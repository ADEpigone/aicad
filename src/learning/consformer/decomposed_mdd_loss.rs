//! The ConsFormer-MDD loss: the weighted model count of each constraint, decomposed into its own
//! exact MDD.
//!
//! For a sample with constraints $c_1, \dots, c_m$ and network probabilities $p$, the loss is
//! $\frac{1}{m}\sum_j - \log(\mathrm{WMC}\_{c_j}(p) + \varepsilon)$, where $\mathrm{WMC}\_{c_j}$ is the
//! weighted model count of the MDD of $c_j$ with the probabilities of the network on its edges.
//! The loss of a batch is the mean over its samples.
//!
//! # Without belief propagation
//!
//! ## The count
//!
//! Let $X_c$ be the scope of $c$ and $q_x$ the probabilities of the network for the variable $x$
//! (one weight $q_x(v)$ per value $v$; a value outside the real domain of $x$ has weight $0$). The
//! weighted model count of $c$ under per-variable weights $w$ is
//!
//! $$\mathrm{WMC}\_c(w) = \sum_{\tau \models c} \ \prod_{x \in X_c} w_x(\tau(x)),$$
//!
//! the sum of the weights of the paths of the MDD of $c$, where the edge of the layer of $x$ with
//! value $v$ has weight $w_x(v)$ (see [`crate::mdd::wmc`]). With $\varepsilon = 10^{-6}$, the loss
//! of $c$ is $L_c = -\ln(\mathrm{WMC}\_c(q) + \varepsilon)$: the labels of the MDD are the
//! probabilities of the network themselves. The $\varepsilon$ keeps the loss finite when no path
//! has a positive weight, for example when the values pinned by the destroy step already violate
//! $c$.
//!
//! ## The gradient
//!
//! Write $W$ for $\mathrm{WMC}\_c$ and fix a variable $x$ of $X_c$. Every path of the MDD crosses
//! the layer of $x$ exactly once, on one edge $e$. Let $\alpha(n)$ be the total weight of the paths
//! from the root to the node $n$, $\beta(n)$ the total weight of the paths from $n$ to the sink, and
//! $v_e$ the value of the edge $e$. The paths through $e$ weigh
//! $\alpha(\mathrm{from}(e)) \ w_x(v_e) \ \beta(\mathrm{to}(e))$ together, so grouping the paths by
//! the edge where they cross the layer of $x$, then the edges by their value,
//!
//! $$W = \sum_{e} \alpha(\mathrm{from}(e)) \ w_x(v_e) \ \beta(\mathrm{to}(e)) = \sum_v w_x(v) \ R_x(v), \qquad R_x(v) = \sum_{e \ :\ v_e = v} \alpha(\mathrm{from}(e)) \ \beta(\mathrm{to}(e)),$$
//!
//! where the sums over $e$ are over the edges of the layer of $x$. $\alpha$ and $\beta$ only
//! involve the labels of the *other* layers, so $R_x$ does not depend on $w_x$, and the count is
//! linear in the labels of $x$:
//!
//! $$\frac{\partial W}{\partial w_x(v)} = R_x(v).$$
//!
//! The loss is $L_c = -\ln(W + \varepsilon)$, so $\partial L_c / \partial W = -1 / (W + \varepsilon)$.
//! The labels are the probabilities, $w_x = q_x$, and by the chain rule
//!
//! $$\frac{\partial L_c}{\partial q_x(v)} = \frac{\partial L_c}{\partial W} \ \frac{\partial W}{\partial w_x(v)} = -\frac{R_x(v)}{W + \varepsilon}.$$
//!
//! Two facts about $R_x$ are used below.
//!
//! - $w_x(v) \ R_x(v)$ is the weight of the paths that give $v$ to $x$, and so
//!   $w_x(v) \ R_x(v) / W$ is the probability that $x = v$ given that $c$ holds, if each variable
//!   is drawn from its labels.
//! - Summing over the values gives back the count: $\sum_v w_x(v) \ R_x(v) = W$.
//!
//! The loss of a sample is the mean over its $m$ constraints, so the gradient for $q_x$ is $1/m$
//! times the sum of these over the constraints that contain $x$; the others do not depend on
//! $q_x$. A value outside the domain of $x$ has a constant label $0$, and no gradient.
//!
//! # With belief propagation (`bp_rounds` $= k > 0$)
//!
//! The labels of the MDD of $c$ are no longer $q$. They are what belief propagation (see
//! [`crate::bp`] for the messages and its notation) would put on the MDD of $c$ after $k$ rounds,
//! with the network as the permanent prior of every variable. The loss then scores $c$ under what
//! the *rest* of the problem believes about its variables. The computation has four steps, for each
//! sample.
//!
//! ## 1. The cavity factors
//!
//! Run $k$ rounds of belief propagation from the priors $q$ (fewer if a tolerance is set with
//! [`DecomposedMddLoss::with_bp_tolerance`] and the marginals settle first, see
//! [`crate::bp::BpRounds`]). Variables that the network does not
//! re-predict are `decided`: their prior is a one-hot on their current value. Let
//! $\mu_{c^\prime \to x}$ be the message of the constraint $c^\prime$ to the variable $x$ in the last round
//! (a likelihood, see [`crate::bp`]). The *cavity factor* of $x$ for $c$ is the evidence about $x$
//! that comes from the constraints *other than $c$*:
//!
//! $$f_{c,x}(v) \ \propto \prod_{c^\prime \ni x,\ c^\prime \neq c} \mu_{c^\prime \to x}(v).$$
//!
//! It is the marginal of $x$ with the prior $q_x$ and the message $\mu_{c \to x}$ of $c$ itself
//! divided out: $f_{c,x} \propto m_x / (q_x \  \mu_{c \to x})$. A variable that is in no other
//! constraint, or no round, gives a uniform factor. Only the shape of $f$ matters, since it is
//! renormalised below. The factors are floored to keep the labels in the range of floating point
//! numbers (see the numerical considerations).
//!
//! ## 2. The labels
//!
//! The label of $x$ in the MDD of $c$ is the cavity message of $x$ to $c$: the prior combined
//! with what the other constraints say,
//!
//! $$\mu_{x \to c}(v) = \frac{q_x(v)\  f_{c,x}(v)}{Z_{c,x}}, \qquad Z_{c,x} = \sum_u q_x(u)\  f_{c,x}(u).$$
//!
//! Let $s_x = \sum_v q_x(v)$ be the mass of the prior: it is $1$ unless some values were masked
//! out of the domain, and keeping it means that probability put outside the domain is still
//! penalised, as it is without belief propagation. The labels are $w_{c,x} = s_x\  \mu_{x \to c}$.
//! With $f$ uniform, $\mu_{x \to c} = q_x / s_x$ and $w_{c,x} = q_x$: a variable that no other
//! constraint speaks about keeps its probabilities.
//!
//! ## 3. The count
//!
//! The loss of $c$ is $L_c = -\ln(\mathrm{WMC}\_c(w_c) + \varepsilon)$ with $w_c = (w_{c,x})\_{x \in
//! X_c}$, which is the probability that $c$ holds if each of its variables is drawn from the
//! network's distribution *tilted* by the evidence of the other constraints. The loss of a sample
//! is the mean of the $L_c$ over its constraints.
//!
//! ## 4. The gradient
//!
//! The loss of $c$ depends on $q_x$ only through the labels $w_{c,x}$ of step 2, and the cavity
//! factor $f_{c,x}$ is **held constant** (see the design notes). In this section $x$ and $c$ are
//! fixed, and the indices are dropped: $q$, $f$, $Z$, $s$, $w$ and $R$ stand for $q_x$, $f_{c,x}$,
//! $Z_{c,x}$, $s_x$, $w_{c,x}$ and $R_x$, and $\mu$ for the message $\mu_{x \to c}$. $W$ and $R$
//! are those of the MDD of $c$ labelled by $w_c$, as in the gradient without belief propagation.
//!
//! ### 4a. The chain rule
//!
//! Moving $q(u)$ moves the labels $w(v)$, which move the count $W$, which moves the loss:
//!
//! $$\frac{\partial L_c}{\partial q(u)} = \sum_v \frac{\partial L_c}{\partial W} \ \frac{\partial W}{\partial w(v)} \ \frac{\partial w(v)}{\partial q(u)} = -\frac{1}{W + \varepsilon} \sum_v R(v) \ \frac{\partial w(v)}{\partial q(u)}.$$
//!
//! The first two factors are those without belief propagation, $-1 / (W + \varepsilon)$ and
//! $R(v)$. The third is new.
//!
//! ### 4b. The derivative of the labels
//!
//! The labels are $w(v) = s \ q(v) \ f(v) / Z$ with $s = \sum_v q(v)$ and
//! $Z = \sum_v q(v) \ f(v)$. Moving $q(u)$ moves three things: the mass, with
//! $\partial s / \partial q(u) = 1$; the numerator $q(v) \ f(v)$, only when $v = u$; and the
//! normaliser, with $\partial Z / \partial q(u) = f(u)$. With $\delta_{vu}$ equal to $1$ if
//! $v = u$ and $0$ otherwise, the product and quotient rules give one term for each:
//!
//! $$\frac{\partial w(v)}{\partial q(u)} = \frac{q(v) \ f(v)}{Z} + \frac{s \ f(v) \ \delta_{vu}}{Z} - \frac{s \ q(v) \ f(v) \ f(u)}{Z^2}.$$
//!
//! The first fraction is $\mu(v)$. In the two others, $f(u) / Z = \mu(u) / q(u)$ (where
//! $q(u) > 0$), and $f(v) \ \delta_{vu} = f(u) \ \delta_{vu}$, so
//!
//! $$\frac{\partial w(v)}{\partial q(u)} = \mu(v) + s \ \frac{\mu(u)}{q(u)} \ \bigl(\delta_{vu} - \mu(v)\bigr).$$
//!
//! The first term is the effect of the mass: all the labels scale with $s$. The second is the
//! effect of the tilt: raising $q(u)$ raises the label of $u$ and, through the normalisation,
//! lowers every label in proportion to $\mu$.
//!
//! ### 4c. The product
//!
//! Putting 4b in 4a,
//!
//! $$\frac{\partial L_c}{\partial q(u)} = -\frac{1}{W + \varepsilon} \sum_v R(v) \Bigl(\mu(v) + s \ \frac{\mu(u)}{q(u)} \ \bigl(\delta_{vu} - \mu(v)\bigr)\Bigr).$$
//!
//! The $\delta_{vu}$ keeps only $v = u$ from its sum, so with $M = \sum_v R(v) \ \mu(v)$,
//!
//! $$\frac{\partial L_c}{\partial q(u)} = -\frac{1}{W + \varepsilon} \Bigl(M + s \ \frac{\mu(u)}{q(u)} \ \bigl(R(u) - M\bigr)\Bigr).$$
//!
//! Since $w = s \ \mu$, the second fact about $R$ gives $M = \frac{1}{s} \sum_v w(v) \ R(v) = W / s$.
//! And $s \ \mu(u) \ R(u) = w(u) \ R(u)$. So
//!
//! $$\frac{\partial L_c}{\partial q(u)} = -\frac{1}{W + \varepsilon} \Bigl(\frac{W}{s} + \frac{w(u) \ R(u) - \mu(u) \ W}{q(u)}\Bigr).$$
//!
//! ### 4d. Reading it with the messages
//!
//! Factor $W$ out of the parenthesis, and let
//!
//! $$\rho = \frac{W}{W + \varepsilon}, \qquad b_c(u) = \frac{w(u) \ R(u)}{W}.$$
//!
//! $\rho$ is close to $1$ unless $c$ is nearly impossible under the labels ($W \lesssim \varepsilon$),
//! and by the first fact about $R$, $b_c(u)$ is the probability that $x = u$ given that $c$
//! holds. This gives
//!
//! $$\frac{\partial L_c}{\partial q_x(u)} = -\rho \Bigl[\frac{1}{s_x} + \frac{b_c(u) - \mu_{x \to c}(u)}{q_x(u)}\Bigr].$$
//!
//! The quantities in it are those of belief propagation. $R(v)$ is the support that $c$ gives to
//! the value $v$, so the message of $c$ to $x$ is $R$ up to a constant, $\mu_{c \to x} \propto R$
//! ([`crate::bp`] normalises it to sum to one). And $w \propto \mu_{x \to c}$. So
//! $b_c \propto \mu_{x \to c} \ \mu_{c \to x}$, and since $b_c$ sums to one,
//!
//! $$b_c(u) = \frac{\mu_{x \to c}(u) \ \mu_{c \to x}(u)}{\sum_v \mu_{x \to c}(v) \ \mu_{c \to x}(v)}:$$
//!
//! the *belief* of $x$ once $c$ is added to what the other constraints say, the product of the
//! incoming messages normalised, as for a marginal in belief propagation.
//!
//! ### 4e. Check: a uniform factor
//!
//! When $f$ is uniform (a single constraint, or no round), $\mu = q / s$ and $w = q$. The formula
//! of 4c becomes $-\bigl(W/s + R(u) - W/s\bigr) / (W + \varepsilon) = -R(u) / (W + \varepsilon)$:
//! the gradient without belief propagation.
//!
//! ### 4f. With respect to the logits
//!
//! The network produces logits $z$, and $q = \mathrm{softmax}(z)$ for a variable that it
//! re-predicts, so $s = 1$. (The logits here are those after the Gumbel noise, scaled by `gumbel_scale`
//! (zero: no noise); if they are divided by a temperature $\tau$ first, the gradient with respect
//! to the raw output gets a factor $1 / \tau$. The noise is a constant of the gradient, so the
//! formula does not depend on the scale.) The Jacobian of the softmax is $\partial q(v) / \partial z(u) = q(v) \ (\delta_{vu} - q(u))$,
//! so the gradient of any function $L$ is
//!
//! $$\frac{\partial L}{\partial z(u)} = \sum_v \frac{\partial L}{\partial q(v)} \ q(v) \ \bigl(\delta_{vu} - q(u)\bigr) = q(u) \ \frac{\partial L}{\partial q(u)} - q(u) \sum_v q(v) \ \frac{\partial L}{\partial q(v)}.$$
//!
//! With the formula of 4d and $s = 1$:
//!
//! - $q(u) \ \partial L_c / \partial q(u) = -\rho \ \bigl(q(u) + b_c(u) - \mu_{x \to c}(u)\bigr)$;
//! - summing this over $u$, and since $q$, $b_c$ and $\mu_{x \to c}$ each sum to one,
//!   $\sum_v q(v) \ \partial L_c / \partial q(v) = -\rho \ (1 + 1 - 1) = -\rho$.
//!
//! The difference of the two is
//!
//! $$\frac{\partial L_c}{\partial z(u)} = -\rho \ \bigl(q(u) + b_c(u) - \mu_{x \to c}(u)\bigr) + \rho \ q(u) = \rho \ \bigl(\mu_{x \to c}(u) - b_c(u)\bigr).$$
//!
//! The $1/s$ term, the same for every value, contributed the $\rho \ q(u)$ that cancels: the
//! softmax does not see it. Another way to get this result: for $s = 1$ the tilt is a shift of
//! the logits, $\mu_{x \to c} = \mathrm{softmax}(z + \ln f)$, and $\rho \ (\mu - b)$ is the usual
//! gradient of the semantic loss at the logits $z + \ln f$.
//!
//! ### What the gradient says
//!
//! - It compares two distributions over the values of $x$: what the rest of the problem believes
//!   ($\mu_{x \to c}$), and what the belief becomes once $c$ is imposed on it ($b_c$). Gradient
//!   descent raises the logit of the values that $c$ makes more likely than the others believed.
//! - Without belief propagation $\mu_{x \to c} = q_x$: this is the usual gradient of the semantic
//!   loss, the difference between the network's distribution and its distribution conditioned on
//!   $c$.
//! - When $c$ has nothing to say about $x$, its message $\mu_{c \to x}$ is flat, $b_c = \mu_{x \to
//!   c}$ and the signal is zero. A value that the other constraints rule out has
//!   $\mu_{x \to c} \approx 0$ and hardly moves.
//! - $\rho$ switches the signal off when $c$ is hopeless ($W \ll \varepsilon$): the loss is flat
//!   at $-\ln \varepsilon$ there.
//!
//! # Training on partial assignments
//!
//! By default every variable with a choice is re-predicted (`mask_fraction` $= 1$). When training
//! samples are made by a destroy operator instead, as local search repairs (see
//! [`with_infeasible_constraints_skipped`](DecomposedMddLoss::with_infeasible_constraints_skipped)),
//! the other variables are *kept* at the values of a random assignment: they are one-hot in the
//! labels and decided in belief propagation, as in the repair of the local search
//! (`crate::local_search::repair`). Every constraint stays in the factor graph, as there.
//!
//! The kept values can leave a constraint $c$ with no completion: every path of its MDD then
//! crosses a kept layer on an edge of weight zero, and $W_c = 0$ whatever the network says. Such
//! a constraint
//!
//! - has no gradient: $\rho = W / (W + \varepsilon) = 0$ in 4f;
//! - sends the uniform message in belief propagation, every $\alpha \beta$ of its layers being
//!   zero, so it drops out of the products of the marginals: it is neutral, at every round, and
//!   at inference too;
//! - but adds the constant $-\ln \varepsilon \approx 13.8$ to the mean.
//!
//! So the mean is taken over the constraints that the kept values leave satisfiable,
//!
//! $$L = \frac{1}{|S|} \sum_{c \in S} -\ln(\mathrm{WMC}_c + \varepsilon), \qquad S = \lbrace c : c \text{ has a solution that agrees with the kept values} \rbrace,$$
//!
//! which changes the scale of the loss and not its gradient direction within a sample: the
//! constraints left out had no gradient. Whether $c \in S$ is decided by a count of $c$ with weight
//! one on every value of a free variable and the one-hot of a kept one, not by $W_c$, which can be
//! zero only because a confident probability was rounded to zero in `f32`. A sample with no
//! satisfiable constraint has a loss and a gradient of zero, and still counts in the mean of the
//! batch. With every variable re-predicted, every satisfiable problem has $S$ equal to all its
//! constraints, and nothing changes.
//!
//! # Numerical considerations
//!
//! The formulas above say what is computed, not how. The implementation differs in the following
//! points, none of which changes the result beyond rounding.
//!
//! - **Scaled counts.** On a large MDD, $\alpha$, $\beta$, $W$ and $R$ can be far below the
//!   smallest `f64`. The forward-backward pass of [`crate::mdd::wmc`] rescales them layer by layer
//!   and returns $\ln W$ and the ratios $g(v) = R(v) / W = \partial \ln W / \partial w(v)$, which
//!   stay in range. The derivative of the loss with respect to a label is therefore computed as
//!   $-R(v) / (W + \varepsilon) = -\rho \ g(v)$, with
//!   $\rho = \exp(\ln W - \ln(W + \varepsilon))$ and $\ln(W + \varepsilon)$ obtained from $\ln W$
//!   by `log_add_exp`. Without belief propagation, $-\rho \ g$ is the gradient (see
//!   `mdd_loss_and_gradient`).
//! - **Messages from ratios.** $R$ is only known through $g$, a constant multiple of it. This is
//!   enough: $\mu_{c \to x} = g / \sum_v g(v)$, and in $b_c$ every constant cancels.
//!   `prior_gradient` computes the formula of 4d from $q$, $\mu_{x \to c}$, $\mu_{c \to x}$ and
//!   $-\rho$.
//! - **Zero probabilities.** $(b_c(u) - \mu_{x \to c}(u)) / q(u)$ is computed as
//!   $\mu_{x \to c}(u) / q(u)$ times $\mu_{c \to x}(u) / \sum_v \mu_{x \to c}(v) \ \mu_{c \to x}(v) - 1$,
//!   with $\mu_{x \to c}(u) / q(u) = f(u) / Z$. This stays defined where $q(u) = 0$, which happens
//!   once a confident softmax is rounded to `f32`.
//! - **Floored factors.** A value that the other constraints rule out has a cavity factor of
//!   exactly zero, or a denormal one. A layer can then lose all its weight and $\ln W$ and its
//!   gradient overflow. Each factor is floored at $10^{-12}$ times its largest entry
//!   ([`FACTOR_FLOOR`]), which keeps the labels in the range of ordinary floating point numbers.
//! - **Degenerate layers.** When no value survives the tilt ($Z = 0$), the labels of the layer are
//!   the prior. When no path of positive weight crosses the layer, the gradient of the layer is
//!   zero, as $\rho = 0$ would give.
//!
//! The tests check `prior_gradient` against the derivative of the labels computed directly (4a and
//! 4b without the messages): they agree to about $10^{-15}$ relative for ordinary probabilities,
//! and to about $10^{-5}$ for near one-hot ones, where the floor of the factors dominates.
//!
//! # Design notes
//!
//! - **The messages are constants.** The factors $f$ are computed from the probabilities
//!   $q$ of all the variables, so $q_y$ also changes the labels that $x$ sees through $f_{c,x}$.
//!   That effect is ignored: the gradient flows from $c$ to the variables of its scope through its
//!   own labels, and not through the rounds of belief propagation. This is cheap, but biased; the
//!   principled alternative is to differentiate through the rounds, as in approximation-aware
//!   training of graphical models (Stoyanov et al., AISTATS 2011).
//! - **Which round.** The factors are made of the beliefs of the *last* round, which were computed
//!   from the labels of the round before: with $k = 1$ the other constraints speak from the plain
//!   priors, and each further round adds one hop of information.
//! - **Nothing to say.** A constraint whose labels give it no path of positive weight sends the
//!   uniform message (see [`crate::bp`]), so its factor tells nothing. With peaked probabilities
//!   (a low temperature of the network) this happens often, and the factors carry little.
//! - **Cost.** A round is a forward-backward pass over each MDD, as the count itself is: a step
//!   costs about $k + 1$ times the passes of the loss without belief propagation, plus a few
//!   operations per layer for the messages.
//!
//! # Example
//!
//! Take $x \neq y$ and $y \neq z$ over $\lbrace 0, 1 \rbrace$ with $q_x = (0.9, 0.1)$,
//! $q_y = (0.5, 0.5)$ and $q_z = (0.2, 0.8)$, and one round. Without belief propagation both
//! counts are $0.5$. With it, $y$ hears from $y \neq z$ the reverse of $q_z$, and from $x \neq y$
//! the reverse of $q_x$:
//!
//! - In $x \neq y$, $f_y = (0.8, 0.2)$ and $Z = 0.5$, so the message of $y$ to the constraint is
//!   $\mu_{y \to c} = (0.8, 0.2)$ and $\mathrm{WMC} = 0.9 \cdot 0.2 + 0.1 \cdot 0.8 = 0.26$. The
//!   message of the constraint to $y$ is $\mu_{c \to y} = (0.1, 0.9)$ ($y = 0$ needs $x = 1$,
//!   which has probability $0.1$). Their overlap is $0.26$, so the belief of $y$ given that the
//!   constraint holds is $b = (0.308, 0.692)$. With $\rho \approx 1$ the gradient with respect to
//!   the logits is $\mu_{y \to c} - b = (0.49, -0.49)$ and with respect to the probabilities
//!   $\partial L / \partial q_y = -\bigl[(1, 1) + (b - \mu_{y \to c}) / q_y\bigr] \approx (-0.02, -1.98)$:
//!   the network is pushed towards $y = 1$, which is what $x \neq y$ needs when $x$ is likely $0$
//!   and $z$ already pulls $y$ towards $0$.
//! - In $y \neq z$, $f_y = (0.1, 0.9)$ and $\mathrm{WMC} = 0.26$ as well. $x$ and $z$ are in one
//!   constraint each, so their labels are their probabilities.
//!
//! # References
//!
//! - The loss on the count of a constraint is the semantic loss of Xu et al., "A Semantic Loss
//!   Function for Deep Learning with Symbolic Knowledge", ICML 2018, here on the MDD of each
//!   constraint. Taking the constraints one by one, as a product of their probabilities, is the
//!   first independence assumption of Ahmed et al., "Semantic Strengthening of Neuro-Symbolic
//!   Learning", AISTATS 2023.
//! - The messages, the cavity and the permanent prior are those of [`crate::bp`]: Pesant, "From
//!   Support Propagation to Belief Propagation in Constraint Programming", JAIR 2019, and Manibod
//!   et al., "Constrained Sequential Inference in Machine Learning Using Constraint Programming",
//!   IJCAI 2025.
//! - Training through approximate inference: Stoyanov, Ropson and Eisner, "Empirical Risk
//!   Minimization of Graphical Model Parameters Given Approximate Inference, Decoding, and Model
//!   Structure", AISTATS 2011.
//!
//! The gradient is computed by hand, by a sink-to-root pass over each diagram
//! ([`crate::mdd::wmc`]), and handed to Burn's autodiff with the stop-gradient identity described
//! at [`DecomposedMddLoss::wmc_loss`]. The diagrams depend only on the problems, so they are
//! compiled once, by [`Loss::prepare`], before training.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use rayon::prelude::*;

use crate::bp::{BpRounds, RoundsStats, cavity_factors_with_rounds};
use crate::learning::Loss;
use crate::mdd::arena::{MddCompilationConfig, compile_constraints};
use crate::mdd::wmc::log_wmc_and_gradient;
use crate::mdd::{CompiledConstraint, MddArena};
use crate::modelling::{Problem, ValueIndex};
use crate::utils::tensor::to_rows;

use super::utils::{blend_with_current, gumbel_softmax};
use crate::neural::consformer::ConsFormerBatch;

/// Added to a weighted model count before its log, to avoid `ln(0)`.
const WMC_EPS: f64 = 1e-6;

/// `ln(exp(a) + exp(b))` without overflow or underflow; `a` may be `-inf`, `b` must be finite.
fn log_add_exp(a: f64, b: f64) -> f64 {
    let largest = a.max(b);
    largest + ((a - largest).exp() + (b - largest).exp()).ln()
}

/// Builds `constraint`'s per-layer weights from the raw network probabilities, **masking out** every
/// raw domain value that isn't actually in `real_problem`'s (possibly narrowed -- e.g. a Sudoku
/// given, or a masked/destroyed variable) domain for that layer's variable.
fn layer_weights_from_probs(
    constraint: &CompiledConstraint,
    real_problem: &Problem,
    probs_for_sample: &[f32],
    domain_size: usize,
) -> Vec<Vec<f64>> {
    let num_layers = constraint.order.len();
    let mut weights: Vec<Vec<f64>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        let variable = constraint.decision_at_layer(layer);
        let mut layer_weights = vec![0.0; domain_size];
        for (raw_value, w) in layer_weights.iter_mut().enumerate() {
            if real_problem[variable].in_domain(raw_value as isize) {
                *w = probs_for_sample[variable.0 * domain_size + raw_value] as f64;
            }
        }
        weights.push(layer_weights);
    }
    weights
}

/// What the belief propagation of a sample needs to know about the variables the network does not
/// re-predict: their value, and whether they are held to it.
struct Evidence {
    /// The current value of each variable.
    assignment: Vec<ValueIndex>,
    /// Whether each variable is pinned to its current value (not re-predicted).
    decided: Vec<bool>,
}

impl Evidence {
    /// No variable is pinned: all of them are re-predicted.
    #[cfg(test)]
    fn free(number_vars: usize) -> Self {
        Self {
            assignment: vec![ValueIndex(0); number_vars],
            decided: vec![false; number_vars],
        }
    }
}

/// Whether `constraint` has a solution that gives every decided variable of `evidence` its value
/// (see the module documentation, *Training on partial assignments*): a count with weight one on
/// every value of the domain of a free variable and the one-hot of a decided one.
fn completable(
    constraint: &CompiledConstraint,
    real_problem: &Problem,
    evidence: &Evidence,
    domain_size: usize,
) -> bool {
    if constraint.order.iter().all(|v| !evidence.decided[v.0]) {
        return true;
    }
    let weights: Vec<Vec<f64>> = constraint
        .order
        .iter()
        .map(|&variable| {
            (0..domain_size)
                .map(|raw_value| {
                    let allowed = real_problem[variable].in_domain(raw_value as isize)
                        && (!evidence.decided[variable.0]
                            || evidence.assignment[variable.0].0 == raw_value);
                    if allowed { 1.0 } else { 0.0 }
                })
                .collect()
        })
        .collect();
    crate::mdd::wmc::wmc(constraint, &weights) > 0.0
}

/// The cavity factors of every constraint of a sample after `rounds` rounds of belief
/// propagation from the probabilities of the network, see [`cavity_factors`]. The probabilities
/// are masked as the weights of the diagrams are (`layer_weights_from_probs`): a value outside the
/// real domain of its variable is impossible. Also returns the number of rounds that ran.
fn sample_cavity_factors(
    constraints: &[CompiledConstraint],
    real_problem: &Problem,
    probs_for_sample: &[f32],
    number_vars: usize,
    domain_size: usize,
    rounds: impl Into<BpRounds>,
    evidence: &Evidence,
) -> (Vec<Vec<Vec<f64>>>, usize) {
    let priors: Vec<Vec<f64>> = (0..number_vars)
        .map(|var| {
            (0..domain_size)
                .map(|raw_value| {
                    if real_problem[crate::modelling::VariableIndex(var)]
                        .in_domain(raw_value as isize)
                    {
                        probs_for_sample[var * domain_size + raw_value] as f64
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();
    cavity_factors_with_rounds(
        constraints,
        &priors,
        &evidence.assignment,
        &evidence.decided,
        rounds,
    )
}

/// Smallest cavity factor, relative to the largest of its layer. A value that the other
/// constraints rule out has a factor of exactly zero (or a denormal one), which would let the
/// weight of a whole diagram collapse and the gradient of its logarithm overflow; the floor keeps
/// the labels, and so the loss, in the range of ordinary floating point numbers.
const FACTOR_FLOOR: f64 = 1e-12;

/// `factor` with no entry below [`FACTOR_FLOOR`] times the largest.
fn floored(factor: &[f64]) -> Vec<f64> {
    let largest = factor.iter().copied().fold(0.0, f64::max);
    factor
        .iter()
        .map(|f| f.max(largest * FACTOR_FLOOR))
        .collect()
}

/// The cavity message of a variable to a constraint, $\mu_{x \to c} \propto q_x \odot f_{c,x}$,
/// together with the ratio $\mu_{x \to c} / q_x$ (which is $f_{c,x}$ over the normaliser, and so
/// is defined even where $q_x$ is zero). `None` when nothing survives the product.
fn cavity_message(prior: &[f64], factor: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
    let factor = floored(factor);
    let total: f64 = prior.iter().zip(&factor).map(|(q, f)| q * f).sum();
    if !(total > 0.0 && total.is_finite()) {
        return None;
    }
    let message = prior
        .iter()
        .zip(&factor)
        .map(|(q, f)| q * f / total)
        .collect();
    let ratio = factor.iter().map(|f| f / total).collect();
    Some((message, ratio))
}

/// The labels of a layer: the cavity message $\mu_{x \to c}$, scaled to the mass of the prior so
/// that a prior that sums to one gives labels that do. When no value survives the factor, the
/// labels are the prior itself.
fn cavity_labels(prior: &[f64], factor: &[f64]) -> Vec<f64> {
    match cavity_message(prior, factor) {
        Some((message, _)) => {
            let mass: f64 = prior.iter().sum();
            message.iter().map(|m| mass * m).collect()
        }
        None => prior.to_vec(),
    }
}

/// The derivative of `chain` times the log count with respect to the prior $q_x$ of a layer, where
/// `count_gradient` is $\partial \ln \mathrm{WMC} / \partial w_{c,x}$ at the labels of
/// [`cavity_labels`] and `chain` is $-\rho$. In terms of the messages,
///
/// $$\frac{\partial L_c}{\partial q_x(u)} = -\rho \Bigl[ \frac{1}{s_x} +
/// \frac{\mu_{x \to c}(u)}{q_x(u)} \Bigl( \frac{\mu_{c \to x}(u)}{\langle \mu_{x \to c},
/// \mu_{c \to x} \rangle} - 1 \Bigr) \Bigr],$$
///
/// where $\mu_{c \to x}$ is the count gradient normalised to sum to one. The cavity factor is
/// held constant, and the layer is treated as the prior itself when no value survives it. Zero
/// when the constraint has no path through the layer.
fn prior_gradient(prior: &[f64], factor: &[f64], count_gradient: &[f64], chain: f64) -> Vec<f64> {
    let mass: f64 = prior.iter().sum();
    let (message, ratio) = match cavity_message(prior, factor) {
        Some(found) => found,
        None if mass > 0.0 && mass.is_finite() => (
            prior.iter().map(|q| q / mass).collect(),
            vec![1.0 / mass; prior.len()],
        ),
        None => return vec![0.0; prior.len()],
    };
    let norm: f64 = count_gradient.iter().sum();
    let to_variable: Vec<f64> = count_gradient.iter().map(|g| g / norm).collect();
    let overlap: f64 = message.iter().zip(&to_variable).map(|(m, c)| m * c).sum();
    if !(norm > 0.0 && norm.is_finite() && overlap > 0.0 && overlap.is_finite()) {
        return vec![0.0; prior.len()];
    }
    ratio
        .iter()
        .zip(&to_variable)
        .map(|(r, c)| chain * (1.0 / mass + r * (c / overlap - 1.0)))
        .collect()
}

/// One constraint MDD's contribution to a sample's loss and to `∂loss/∂probs_for_sample`, computed
/// by a plain sink-to-root chain-rule pass (`crate::mdd::wmc::wmc_and_gradient`) instead of
/// automatic differentiation.
/// `probs_for_sample` is that sample's flattened `(number_vars, domain_size)` probability slice,
/// indexed by each variable's *raw domain value*, not by `ValueIndex` position.
/// `grad` accumulates `∂(-log(wmc+eps))/∂probs_for_sample` into that same indexing.
/// With `factors` (one cavity factor per layer, see [`cavity_factors`]) the diagram is labelled
/// with the cavity messages of the variables (scaled to the mass of their priors), and the gradient
/// goes back to the probabilities through them.
fn mdd_loss_and_gradient(
    constraint: &CompiledConstraint,
    real_problem: &Problem,
    probs_for_sample: &[f32],
    domain_size: usize,
    factors: Option<&[Vec<f64>]>,
    grad: &mut [f32],
) -> f64 {
    let prior = layer_weights_from_probs(constraint, real_problem, probs_for_sample, domain_size);
    let weights: Vec<Vec<f64>> = match factors {
        Some(factors) => prior
            .iter()
            .zip(factors)
            .map(|(w, f)| cavity_labels(w, f))
            .collect(),
        None => prior.clone(),
    };
    let (log_wmc, gradient) = log_wmc_and_gradient(constraint, &weights);
    // `ln(wmc + eps)` and `d(-ln(wmc + eps))/d(weight) = -wmc / (wmc + eps) * d(ln wmc)/d(weight)`,
    // computed from `ln wmc` so that a count below the smallest `f64` does not flatten the loss.
    let log_total = log_add_exp(log_wmc, WMC_EPS.ln());
    let chain = -(log_wmc - log_total).exp();

    for (layer, layer_gradient) in gradient.iter().enumerate() {
        let variable = constraint.decision_at_layer(layer);
        let layer_gradient: Vec<f64> = match factors {
            Some(factors) => prior_gradient(&prior[layer], &factors[layer], layer_gradient, chain),
            None => layer_gradient.iter().map(|g| chain * g).collect(),
        };
        for (raw_value, &g) in layer_gradient.iter().enumerate() {
            // Masked the same way `layer_weights_from_probs` masked the forward weight: a raw
            // value outside `real_problem`'s actual domain has a *constant* (zero) weight,
            // independent of `probs_for_sample` -- so its contribution to d(loss)/d(probs) is
            // genuinely zero, not whatever d(wmc)/d(weight) the shared, unhinted structure
            // reports for that (structurally still-present, but never actually selectable) edge.
            if real_problem[variable].in_domain(raw_value as isize) {
                grad[variable.0 * domain_size + raw_value] += g as f32;
            }
        }
    }

    -log_total
}

/// A sample's plain average of `-log(wmc+eps)` over its constraint MDDs, plus
/// `∂(that average)/∂probs` (`mdds` iterated sequentially -- parallelism is across samples, see
/// `mdd_wmc_loss`).
///
/// With `bp_rounds` allowing a round or more, the diagrams are labelled after the rounds of belief
/// propagation that run (see the module documentation); `evidence` says which variables are pinned.
/// With `skip_infeasible`, the mean is over the constraints that the pinned values leave
/// satisfiable (see *Training on partial assignments*), instead of over all of them.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn sample_loss_and_gradient(
    constraints: &[CompiledConstraint],
    real_problem: &Problem,
    probs_for_sample: &[f32],
    number_vars: usize,
    domain_size: usize,
    bp_rounds: impl Into<BpRounds>,
    evidence: &Evidence,
    skip_infeasible: bool,
) -> (f64, Vec<f32>) {
    let (loss, grad, _) = sample_loss_gradient_and_rounds(
        constraints,
        real_problem,
        probs_for_sample,
        number_vars,
        domain_size,
        bp_rounds,
        evidence,
        skip_infeasible,
    );
    (loss, grad)
}

/// [`sample_loss_and_gradient`], and the number of rounds of belief propagation that ran, `None`
/// without belief propagation.
#[allow(clippy::too_many_arguments)]
fn sample_loss_gradient_and_rounds(
    constraints: &[CompiledConstraint],
    real_problem: &Problem,
    probs_for_sample: &[f32],
    number_vars: usize,
    domain_size: usize,
    bp_rounds: impl Into<BpRounds>,
    evidence: &Evidence,
    skip_infeasible: bool,
) -> (f64, Vec<f32>, Option<usize>) {
    let bp_rounds = bp_rounds.into();
    let factors_and_rounds = (bp_rounds.max > 0).then(|| {
        sample_cavity_factors(
            constraints,
            real_problem,
            probs_for_sample,
            number_vars,
            domain_size,
            bp_rounds,
            evidence,
        )
    });
    let (factors, rounds_run) = match factors_and_rounds {
        Some((factors, rounds)) => (Some(factors), Some(rounds)),
        None => (None, None),
    };
    let mut grad = vec![0.0f32; number_vars * domain_size];
    let mut loss_sum = 0.0;
    let mut counted = 0usize;
    for (index, constraint) in constraints.iter().enumerate() {
        if skip_infeasible && !completable(constraint, real_problem, evidence, domain_size) {
            continue;
        }
        counted += 1;
        loss_sum += mdd_loss_and_gradient(
            constraint,
            real_problem,
            probs_for_sample,
            domain_size,
            factors.as_ref().map(|f| f[index].as_slice()),
            &mut grad,
        );
    }
    // `.max(1)` so a (degenerate) sample with zero constraints reads back as zero loss instead of
    // `0.0 / 0.0`, matching the previous floor-based behaviour's result in that case.
    let denom = counted.max(1) as f64;
    let loss = loss_sum / denom;
    for g in &mut grad {
        *g /= denom as f32;
    }
    (loss, grad, rounds_run)
}

/// Plain (unweighted) per-constraint mean of `-log(wmc+eps)`, per-sample then per-batch -- see
/// `sample_loss_and_gradient`.
///
/// The loss decomposes each constraint of a problem into its own exact MDD, so it needs those
/// diagrams before it can score anything: [`Loss::prepare`] compiles them for every problem it
/// will be asked about, and keeps them (with the arena they were compiled in) until the next call.
pub struct DecomposedMddLoss {
    /// How the constraint MDDs are compiled. Always exact, whatever width the setting asked for.
    compilation: MddCompilationConfig,
    /// Number of values the network predicts per variable; the MDDs are compiled against it.
    domain_size: usize,
    /// Rounds of belief propagation that label the diagrams before the loss is computed; none
    /// scores the probabilities of the network as they are.
    bp_rounds: usize,
    /// Belief propagation stops before `bp_rounds` once no marginal moves by this much in a round
    /// (see [`BpRounds`]); zero runs every round.
    bp_tolerance: f64,
    /// Multiplier of the Gumbel(0, 1) noise added to the logits before their softmax: `1.0` is the
    /// Gumbel-softmax relaxation, `0.0` no noise (and a deterministic loss).
    gumbel_scale: f32,
    /// Whether the mean of a sample is over the constraints its kept values leave satisfiable
    /// (see the module documentation), instead of over all of them.
    skip_infeasible: bool,
    /// The rounds of belief propagation run per sample since the last [`Loss::take_bp_rounds`].
    rounds_run: Mutex<RoundsStats>,
    /// Where the MDDs are compiled, so that constraints of the same shape share their structure.
    arena: MddArena,
    /// The problems compiled by the last `compile`; they keep alive the addresses `compiled` is
    /// keyed by.
    problems: Vec<Arc<Problem>>,
    /// The constraints of each problem, one `CompiledConstraint` per constraint, keyed by the
    /// address of the problem's `Arc`.
    compiled: HashMap<usize, Vec<CompiledConstraint>>,
}

impl DecomposedMddLoss {
    /// A loss for a network predicting `domain_size` values per variable. `compilation` chooses
    /// the variable ordering and the heuristics; its width bound and iteration limit are ignored,
    /// since the loss needs exact diagrams.
    pub fn new(compilation: MddCompilationConfig, domain_size: usize) -> Self {
        Self {
            compilation: MddCompilationConfig {
                max_width: usize::MAX,
                max_iterations: usize::MAX,
                ..compilation
            },
            domain_size,
            bp_rounds: 0,
            bp_tolerance: 0.0,
            gumbel_scale: 1.0,
            skip_infeasible: false,
            rounds_run: Mutex::new(RoundsStats::default()),
            arena: MddArena::new(),
            problems: Vec::new(),
            compiled: HashMap::new(),
        }
    }

    /// Labels the diagram of each constraint with the probabilities of the network after
    /// `rounds` rounds of belief propagation, instead of with the probabilities themselves. See the
    /// module documentation. Zero, the default, is no belief propagation.
    pub fn with_bp_rounds(mut self, rounds: usize) -> Self {
        self.bp_rounds = rounds;
        self
    }

    /// Stops belief propagation before its `bp_rounds` rounds once no marginal moves by
    /// `tolerance` or more in a round, in total variation distance (see [`BpRounds`]); the
    /// factors are then those of the last round that ran. Zero, the default, runs every round.
    ///
    /// # Panics
    ///
    /// If `tolerance` is negative or not finite.
    pub fn with_bp_tolerance(mut self, tolerance: f64) -> Self {
        self.bp_tolerance = BpRounds::until_converged(0, tolerance).tolerance;
        self
    }

    /// Sets the scale of the Gumbel noise added to the network's logits before the softmax that
    /// turns them into the probabilities the loss scores. One, the default, is the Gumbel-softmax
    /// relaxation of ConsFormer; zero adds no noise, so the probabilities are the plain softmax of
    /// the logits and the loss is deterministic. The noise is added before both the labels of
    /// the diagrams and the belief propagation read the probabilities, so the scale applies to
    /// the whole loss (see the module documentation, 4f).
    ///
    /// # Panics
    ///
    /// If `scale` is negative or not finite.
    pub fn with_gumbel_scale(mut self, scale: f32) -> Self {
        assert!(
            scale.is_finite() && scale >= 0.0,
            "the Gumbel scale must be finite and non-negative, got {scale}"
        );
        self.gumbel_scale = scale;
        self
    }

    /// Takes the mean of each sample over the constraints that its kept variables (those it does
    /// not re-predict) leave satisfiable, instead of over all of them; the default, `false`, counts
    /// every constraint. A constraint left out had no gradient and only added $-\ln \varepsilon$
    /// to the loss, and every constraint stays in belief propagation, as at inference. Meant for
    /// samples made by a destroy operator, whose kept variables hold the values of a random
    /// assignment: see the module documentation. When every variable is re-predicted the two give
    /// the same loss.
    pub fn with_infeasible_constraints_skipped(mut self, skip: bool) -> Self {
        self.skip_infeasible = skip;
        self
    }

    /// Compiles the constraints of `problems`, replacing those of the previous call. A problem
    /// given several times (the same `Arc`) is compiled once. Every problem the loss will score,
    /// training and validation alike, must be given.
    ///
    /// Problems are known by their identity: two equal problems held in different `Arc`s are
    /// compiled separately.
    fn compile(&mut self, problems: &[Arc<Problem>]) {
        let mut seen = HashSet::new();
        let unique: Vec<Arc<Problem>> = problems
            .iter()
            .filter(|problem| seen.insert(Self::key(problem)))
            .cloned()
            .collect();

        let progress = ProgressBar::new(unique.len() as u64);
        progress.set_style(
            ProgressStyle::with_template(
                "{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .expect("hard-coded progress bar template should always be valid"),
        );
        progress.set_message("Compiling MDDs");

        let start = Instant::now();
        let compiled: Vec<Vec<CompiledConstraint>> = crate::utils::worker_pool().install(|| {
            unique
                .par_iter()
                .progress_with(progress.clone())
                .map(|problem| {
                    compile_constraints(problem, &self.arena, &self.compilation, self.domain_size)
                })
                .collect()
        });
        progress.finish_and_clear();
        crate::diagnostics::data_log!(
            "mdd_compilation",
            num_problems = unique.len(),
            elapsed_s = start.elapsed().as_secs_f64(),
        );

        self.compiled = unique.iter().map(Self::key).zip(compiled).collect();
        self.problems = unique;
    }

    /// Identifies a problem by the address of its `Arc`.
    fn key(problem: &Arc<Problem>) -> usize {
        Arc::as_ptr(problem) as usize
    }

    /// The compiled constraints of `problem`.
    ///
    /// # Panics
    ///
    /// If `problem` is not (the same `Arc` as) one of the problems of the last `compile`.
    fn compiled_for(&self, problem: &Arc<Problem>) -> &[CompiledConstraint] {
        match self.compiled.get(&Self::key(problem)) {
            Some(constraints) => constraints,
            None => panic!(
                "DecomposedMddLoss was given a problem it has not compiled: `prepare` must be \
                 called with every training and validation problem"
            ),
        }
    }
}

impl DecomposedMddLoss {
    /// Computes the MDD-WMC loss and its exact gradient wrt `probs` by hand (see
    /// `mdd_loss_and_gradient`), then hands that gradient to Burn's autodiff via the standard
    /// stop-gradient identity `value + stop_gradient(desired_value - value)`, here as
    /// `grad_tensor * (probs - probs.detach())`: its *value* is always zero (`probs.clone()` and
    /// `probs.detach()` hold the same numbers), so the returned tensor's value is exactly
    /// `loss_value`, but its derivative wrt `probs` is exactly `grad_tensor`, since `grad_tensor` and
    /// `probs.detach()` are constants as far as autodiff is concerned. This keeps the whole hand-rolled
    /// backward pass outside Burn's graph -- nothing per-layer gets recorded or replayed -- while
    /// still letting `loss.backward()` reach every parameter upstream of `probs` normally.
    fn wmc_loss<B: Backend>(
        &self,
        probs: Tensor<B, 3>,
        batch: &ConsFormerBatch<B>,
    ) -> Tensor<B, 1> {
        let device = probs.device();
        let [batch_size, number_vars, domain_size] = probs.dims();

        let probs_data: Vec<f32> = probs
            .clone()
            .into_data()
            .to_vec::<f32>()
            .expect("probs should convert to f32");

        let compiled: Vec<&[CompiledConstraint]> = batch
            .problems
            .iter()
            .map(|problem| self.compiled_for(problem))
            .collect();

        // The variables the network does not re-predict are evidence for belief propagation, and
        // say which constraints are still satisfiable.
        let evidence: Vec<Evidence> = if self.bp_rounds > 0 || self.skip_infeasible {
            let values = to_rows(&batch.assignments, batch_size, number_vars);
            // Through a float tensor: the storage of a bool tensor depends on the backend (a u8 on
            // CUDA, a native bool on ndarray), and `to_vec::<bool>` only accepts the latter.
            let masks: Vec<bool> = batch
                .var_masks
                .clone()
                .float()
                .into_data()
                .to_vec::<f32>()
                .expect("variable masks should convert to f32")
                .into_iter()
                .map(|m| m > 0.5)
                .collect();
            values
                .iter()
                .enumerate()
                .map(|(i, row)| Evidence {
                    assignment: row.iter().map(|&v| ValueIndex(v.max(0) as usize)).collect(),
                    decided: masks[i * number_vars..(i + 1) * number_vars]
                        .iter()
                        .map(|&re_predicted| !re_predicted)
                        .collect(),
                })
                .collect()
        } else {
            Vec::new()
        };

        let results: Vec<(f64, Vec<f32>, Option<usize>)> =
            crate::utils::worker_pool().install(|| {
                compiled
                    .par_iter()
                    .zip(batch.problems.par_iter())
                    .enumerate()
                    .map(|(i, (constraints, real_problem))| {
                        let start = i * number_vars * domain_size;
                        let sample_probs = &probs_data[start..start + number_vars * domain_size];
                        let no_evidence;
                        let evidence = match evidence.get(i) {
                            Some(evidence) => evidence,
                            None => {
                                no_evidence = Evidence {
                                    assignment: Vec::new(),
                                    decided: Vec::new(),
                                };
                                &no_evidence
                            }
                        };
                        sample_loss_gradient_and_rounds(
                            constraints,
                            real_problem,
                            sample_probs,
                            number_vars,
                            domain_size,
                            BpRounds::until_converged(self.bp_rounds, self.bp_tolerance),
                            evidence,
                            self.skip_infeasible,
                        )
                    })
                    .collect()
            });

        {
            let mut rounds_run = self
                .rounds_run
                .lock()
                .expect("the rounds statistics lock is poisoned");
            for rounds in results.iter().filter_map(|(_, _, rounds)| *rounds) {
                rounds_run.record(rounds);
            }
        }

        let loss_value: f64 =
            results.iter().map(|(loss, _, _)| loss).sum::<f64>() / batch_size as f64;

        let mut grad_data = vec![0.0f32; batch_size * number_vars * domain_size];
        for (i, (_, grad, _)) in results.iter().enumerate() {
            let start = i * number_vars * domain_size;
            for (j, &g) in grad.iter().enumerate() {
                grad_data[start + j] = g / batch_size as f32;
            }
        }

        let grad_tensor: Tensor<B, 3> = Tensor::<B, 1>::from_data(grad_data.as_slice(), &device)
            .reshape([batch_size, number_vars, domain_size]);
        let loss_tensor: Tensor<B, 1> = Tensor::from_data([loss_value as f32].as_slice(), &device);

        (grad_tensor * (probs.clone() - probs.detach()))
            .sum()
            .reshape([1])
            + loss_tensor
    }
}

impl<B: Backend> Loss<B, ConsFormerBatch<B>> for DecomposedMddLoss {
    fn prepare(&mut self, problems: &[Arc<Problem>]) {
        self.compile(problems);
    }

    fn take_bp_rounds(&self) -> RoundsStats {
        std::mem::take(
            &mut *self
                .rounds_run
                .lock()
                .expect("the rounds statistics lock is poisoned"),
        )
    }

    fn loss(&self, logits: Tensor<B, 3>, batch: &ConsFormerBatch<B>) -> Tensor<B, 1> {
        let probs = gumbel_softmax(logits, self.gumbel_scale);
        let probs = blend_with_current(probs, batch.assignments.clone(), batch.var_masks.clone());
        self.wmc_loss(probs, batch)
    }
}

#[cfg(test)]
mod compile_tests {
    use super::*;
    use crate::constraints::{all_different, among, not_equals, sum};

    /// `real_domain_size` is what `problem[variable].domain_size()` used to be before the arena
    /// split -- since the group's structure is now compiled against the full nominal
    /// `domain_size` regardless of any real narrowing, this builds the same masked weights
    /// `crate::learning::consformer::loss::layer_weights_from_probs` does (every real variable
    /// here has an unnarrowed `0..domain_size` domain, so the mask is a no-op, but this keeps the
    /// test helper honest about what production code actually builds).
    fn brute_force_wmc(
        constraint: &CompiledConstraint,
        real_problem: &Problem,
        probs: &[f64],
        domain_size: usize,
    ) -> f64 {
        let num_layers = constraint.order.len();
        let mut weights = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            let variable = constraint.decision_at_layer(layer);
            let mut layer_weights = vec![0.0; domain_size];
            for (raw_value, w) in layer_weights.iter_mut().enumerate() {
                if real_problem[variable].in_domain(raw_value as isize) {
                    *w = probs[variable.0 * domain_size + raw_value];
                }
            }
            weights.push(layer_weights);
        }
        crate::mdd::wmc::wmc(constraint, &weights)
    }

    #[test]
    fn compile_constraints_matches_brute_force_for_not_equals() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2]);
        let y = problem.add_variable(vec![0, 1, 2]);
        not_equals(&mut problem, x, y);
        let problem = Arc::new(problem);

        let domain_size = 3;
        let arena = MddArena::new();
        let mdds = compile_constraints(
            &problem,
            &arena,
            &MddCompilationConfig::default(),
            domain_size,
        );
        assert_eq!(mdds.len(), 1);
        assert_eq!(mdds[0].number_layers() - 1, 2);

        let probs: Vec<f64> = vec![
            0.2, 0.5, 0.3, // x
            0.1, 0.3, 0.6, // y
        ];
        let wmc = brute_force_wmc(&mdds[0], &problem, &probs, domain_size);

        let mut brute = 0.0;
        for xv in 0..domain_size {
            for yv in 0..domain_size {
                if xv != yv {
                    brute += probs[xv] * probs[domain_size + yv];
                }
            }
        }
        assert!((wmc - brute).abs() < 1e-9, "wmc={wmc} brute={brute}");
    }

    #[test]
    fn compile_constraints_matches_brute_force_for_all_different_permutation() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2]);
        all_different(&mut problem, vars.clone());
        let problem = Arc::new(problem);

        let domain_size = 3;
        let arena = MddArena::new();
        let mdds = compile_constraints(
            &problem,
            &arena,
            &MddCompilationConfig::default(),
            domain_size,
        );
        assert_eq!(mdds.len(), 1);
        assert_eq!(mdds[0].number_layers() - 1, 3);

        let probs: Vec<f64> = vec![
            0.5, 0.3, 0.2, // var 0
            0.2, 0.2, 0.6, // var 1
            0.1, 0.4, 0.5, // var 2
        ];
        let wmc = brute_force_wmc(&mdds[0], &problem, &probs, domain_size);

        let mut brute = 0.0;
        for a in 0..domain_size {
            for b in 0..domain_size {
                for c in 0..domain_size {
                    if a != b && b != c && a != c {
                        brute += probs[a] * probs[domain_size + b] * probs[2 * domain_size + c];
                    }
                }
            }
        }
        assert!((wmc - brute).abs() < 1e-9, "wmc={wmc} brute={brute}");
    }

    #[test]
    fn compile_constraints_matches_brute_force_for_sum() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2]);
        sum(&mut problem, vars.clone(), 3);
        let problem = Arc::new(problem);

        let domain_size = 3;
        let arena = MddArena::new();
        let mdds = compile_constraints(
            &problem,
            &arena,
            &MddCompilationConfig::default(),
            domain_size,
        );

        let probs: Vec<f64> = vec![
            0.5, 0.3, 0.2, // var 0
            0.2, 0.2, 0.6, // var 1
            0.1, 0.4, 0.5, // var 2
        ];
        let wmc = brute_force_wmc(&mdds[0], &problem, &probs, domain_size);

        let mut brute = 0.0;
        for a in 0..domain_size {
            for b in 0..domain_size {
                for c in 0..domain_size {
                    if a + b + c == 3 {
                        brute += probs[a] * probs[domain_size + b] * probs[2 * domain_size + c];
                    }
                }
            }
        }
        assert!((wmc - brute).abs() < 1e-9, "wmc={wmc} brute={brute}");
    }

    #[test]
    fn compile_constraints_matches_brute_force_for_among() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2]);
        among(&mut problem, vars.clone(), vec![1, 2], 1, 2);
        let problem = Arc::new(problem);

        let domain_size = 3;
        let arena = MddArena::new();
        let mdds = compile_constraints(
            &problem,
            &arena,
            &MddCompilationConfig::default(),
            domain_size,
        );

        let probs: Vec<f64> = vec![
            0.5, 0.3, 0.2, // var 0
            0.2, 0.2, 0.6, // var 1
            0.1, 0.4, 0.5, // var 2
        ];
        let wmc = brute_force_wmc(&mdds[0], &problem, &probs, domain_size);

        let mut brute = 0.0;
        for a in 0..domain_size {
            for b in 0..domain_size {
                for c in 0..domain_size {
                    let count = [a, b, c].iter().filter(|&&v| v == 1 || v == 2).count();
                    if (1..=2).contains(&count) {
                        brute += probs[a] * probs[domain_size + b] * probs[2 * domain_size + c];
                    }
                }
            }
        }
        assert!((wmc - brute).abs() < 1e-9, "wmc={wmc} brute={brute}");
    }

    #[test]
    fn unsat_constraint_gets_zero_wmc() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1]);
        all_different(&mut problem, vars.clone());
        let problem = Arc::new(problem);

        let domain_size = 2;
        let arena = MddArena::new();
        let mdds = compile_constraints(
            &problem,
            &arena,
            &MddCompilationConfig::default(),
            domain_size,
        );
        let probs: Vec<f64> = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        assert_eq!(
            brute_force_wmc(&mdds[0], &problem, &probs, domain_size),
            0.0
        );
    }

    #[test]
    #[should_panic(expected = "out of the nominal")]
    fn compile_constraints_panics_when_domain_size_is_too_small() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2]);
        let y = problem.add_variable(vec![0, 1, 2]);
        not_equals(&mut problem, x, y);
        let problem = Arc::new(problem);

        let arena = MddArena::new();
        compile_constraints(&problem, &arena, &MddCompilationConfig::default(), 2);
    }

    #[test]
    fn compiling_many_problems_matches_compiling_each_alone() {
        let mut problems = Vec::new();
        for domain_max in 0isize..40 {
            let mut problem = Problem::default();
            let domain: Vec<isize> = (0..=(domain_max % 5 + 2)).collect();
            let vars = problem.add_variables(3, domain);
            all_different(&mut problem, vars.clone());
            not_equals(&mut problem, vars[0], vars[1]);
            problems.push(Arc::new(problem));
        }
        let domain_size = 7;

        let mut loss_fn = DecomposedMddLoss::new(MddCompilationConfig::default(), domain_size);
        loss_fn.compile(&problems);

        assert_eq!(loss_fn.compiled.len(), problems.len());
        for problem in &problems {
            let expected = compile_constraints(
                problem,
                &MddArena::new(),
                &MddCompilationConfig::default(),
                domain_size,
            );
            let actual = loss_fn.compiled_for(problem);
            assert_eq!(actual.len(), expected.len());
            for (a, e) in actual.iter().zip(expected.iter()) {
                assert_eq!(a.number_layers(), e.number_layers());
                assert_eq!(a.number_nodes(), e.number_nodes());
                assert_eq!(a.number_edges(), e.number_edges());
            }
        }
    }

    #[test]
    fn a_problem_given_twice_is_compiled_once() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2]);
        all_different(&mut problem, vars);
        let problem = Arc::new(problem);

        let mut loss_fn = DecomposedMddLoss::new(MddCompilationConfig::default(), 3);
        loss_fn.compile(&[Arc::clone(&problem), Arc::clone(&problem)]);
        assert_eq!(loss_fn.compiled.len(), 1);
        assert_eq!(loss_fn.compiled_for(&problem).len(), 1);
    }

    #[test]
    #[should_panic(expected = "has not compiled")]
    fn scoring_a_problem_that_was_not_prepared_panics() {
        let build = || {
            let mut problem = Problem::default();
            let x = problem.add_variable(vec![0, 1, 2]);
            let y = problem.add_variable(vec![0, 1, 2]);
            not_equals(&mut problem, x, y);
            Arc::new(problem)
        };
        let mut loss_fn = DecomposedMddLoss::new(MddCompilationConfig::default(), 3);
        loss_fn.compile(&[build()]);
        // Equal to the compiled problem, but another `Arc`.
        loss_fn.compiled_for(&build());
    }
}

#[cfg(test)]
mod loss_tests {
    use std::sync::Arc;

    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::data::dataloader::batcher::Batcher;
    use burn::data::dataset::Dataset;

    use crate::constraints::{all_different, not_equals};
    use crate::mdd::MddArena;
    use crate::mdd::arena::MddCompilationConfig;
    use crate::modelling::Problem;

    use burn::tensor::Int;

    use super::*;
    use crate::neural::consformer::{ConsFormerBatcher, ConsFormerDataset};

    /// Every problem is 3 variables, domain `{0,1,2}`, with an `AllDifferent` over all three and
    /// a `NotEquals` over the first two. Generic over the backend so the same builder can be used
    /// both with plain `NdArray` (the WMC/loss correctness tests) and `Autodiff<NdArray>` (the
    /// gradient-flow tests).
    fn two_sample_batch<B: Backend>(
        device: &B::Device,
        mask_fraction: f64,
    ) -> (ConsFormerBatch<B>, DecomposedMddLoss) {
        let mut problems = Vec::new();
        for _ in 0..2 {
            let mut problem = Problem::default();
            let vars = problem.add_variables(3, vec![0, 1, 2]);
            all_different(&mut problem, vars.clone());
            not_equals(&mut problem, vars[0], vars[1]);
            problems.push(Arc::new(problem));
        }

        let mut loss_fn = DecomposedMddLoss::new(MddCompilationConfig::default(), 3);
        loss_fn.compile(&problems);
        let dataset = ConsFormerDataset::<B>::new(problems, device);
        let samples: Vec<_> = (0..dataset.len())
            .map(|i| dataset.get(i).unwrap())
            .collect();

        let batcher = ConsFormerBatcher {
            mask_fraction,
            destroy: None,
        };
        (batcher.batch(samples, device), loss_fn)
    }

    fn brute_force_not_equals(probs: &[f64], domain_size: usize) -> f64 {
        let mut brute = 0.0;
        for a in 0..domain_size {
            for b in 0..domain_size {
                if a != b {
                    brute += probs[a] * probs[domain_size + b];
                }
            }
        }
        brute
    }

    fn brute_force_all_different_permutation(probs: &[f64], domain_size: usize) -> f64 {
        let mut brute = 0.0;
        for a in 0..domain_size {
            for b in 0..domain_size {
                for c in 0..domain_size {
                    if a != b && b != c && a != c {
                        brute += probs[a] * probs[domain_size + b] * probs[2 * domain_size + c];
                    }
                }
            }
        }
        brute
    }

    /// Plain per-constraint mean of `-log(wmc+eps)`, matching `sample_loss_and_gradient`'s (now
    /// unweighted) reduction -- used by the tests below to build an independently-derived expected
    /// value from brute-force/hand-computed `wmc`s.
    fn plain_average(wmcs: &[f64]) -> f64 {
        let loss_sum: f64 = wmcs.iter().map(|&wmc| -(wmc + WMC_EPS).ln()).sum();
        loss_sum / (wmcs.len().max(1) as f64)
    }

    #[test]
    fn sample_loss_and_gradient_matches_brute_force_average() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2]);
        all_different(&mut problem, vars.clone());
        not_equals(&mut problem, vars[0], vars[1]);
        let problem = Arc::new(problem);

        let mut loss_fn = DecomposedMddLoss::new(MddCompilationConfig::default(), 3);
        loss_fn.compile(&[Arc::clone(&problem)]);
        let mdds = loss_fn.compiled_for(&problem);

        let probs: Vec<f32> = vec![0.2, 0.5, 0.3, 0.1, 0.3, 0.6, 0.4, 0.4, 0.2];
        let (loss, _grad) =
            sample_loss_and_gradient(mdds, &problem, &probs, 3, 3, 0, &Evidence::free(3), false);

        let probs_f64: Vec<f64> = probs.iter().map(|&v| v as f64).collect();
        let not_equals_wmc = brute_force_not_equals(&probs_f64, 3);
        let all_different_wmc = brute_force_all_different_permutation(&probs_f64, 3);
        let expected = plain_average(&[not_equals_wmc, all_different_wmc]);

        assert!(
            (loss - expected).abs() < 1e-6,
            "got {loss} expected {expected}"
        );
    }

    #[test]
    fn sample_loss_and_gradient_matches_finite_differences() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2]);
        all_different(&mut problem, vars.clone());
        not_equals(&mut problem, vars[0], vars[1]);
        let problem = Arc::new(problem);

        let mut loss_fn = DecomposedMddLoss::new(MddCompilationConfig::default(), 3);
        loss_fn.compile(&[Arc::clone(&problem)]);
        let mdds = loss_fn.compiled_for(&problem);

        let probs: Vec<f32> = vec![0.2, 0.5, 0.3, 0.1, 0.3, 0.6, 0.4, 0.4, 0.2];
        let (base_loss, grad) =
            sample_loss_and_gradient(mdds, &problem, &probs, 3, 3, 0, &Evidence::free(3), false);

        let eps = 1e-4f32;
        for i in 0..probs.len() {
            let mut bumped = probs.clone();
            bumped[i] += eps;
            let (bumped_loss, _) = sample_loss_and_gradient(
                mdds,
                &problem,
                &bumped,
                3,
                3,
                0,
                &Evidence::free(3),
                false,
            );
            let finite_diff = (bumped_loss - base_loss) / eps as f64;
            assert!(
                (finite_diff - grad[i] as f64).abs() < 1e-2,
                "index {i}: analytic={} finite_diff={finite_diff}",
                grad[i],
            );
        }
    }

    #[test]
    fn mdd_wmc_loss_averages_per_sample_then_per_batch() {
        let device = NdArrayDevice::default();
        let domain_size = 3;
        let (batch, loss_fn) = two_sample_batch::<NdArray>(&device, 0.0);

        let per_sample_probs: Vec<Vec<f64>> = vec![
            vec![0.2, 0.5, 0.3, 0.1, 0.3, 0.6, 0.4, 0.4, 0.2],
            vec![0.6, 0.3, 0.1, 0.2, 0.2, 0.6, 0.5, 0.25, 0.25],
        ];
        let flat: Vec<f32> = per_sample_probs
            .iter()
            .flatten()
            .map(|&v| v as f32)
            .collect();
        let probs: Tensor<NdArray, 3> =
            Tensor::<NdArray, 1>::from_data(flat.as_slice(), &device).reshape([2, 3, domain_size]);

        let loss = loss_fn.wmc_loss(probs, &batch);
        let loss_value: f32 = loss.into_data().to_vec::<f32>().unwrap()[0];

        let mut expected_per_sample = Vec::new();
        for probs in &per_sample_probs {
            let not_equals_wmc = brute_force_not_equals(probs, domain_size);
            let all_different_wmc = brute_force_all_different_permutation(probs, domain_size);
            expected_per_sample.push(plain_average(&[not_equals_wmc, all_different_wmc]));
        }
        let expected = expected_per_sample.iter().sum::<f64>() / expected_per_sample.len() as f64;

        assert!(
            (loss_value as f64 - expected).abs() < 1e-4,
            "got {loss_value} expected {expected}"
        );
    }

    /// The custom gradient this loss hands to Burn's autodiff (`mdd_loss_and_gradient`'s
    /// sink-to-root chain rule) bypasses Burn's own backward machinery entirely, so this checks it
    /// against finite differences of `mdd_wmc_loss` itself -- computed on a plain, non-autodiff
    /// `NdArray` backend at perturbed `probs` -- rather than against a second differentiation path
    /// through Burn. `mdd_wmc_loss` is deterministic (no `gumbel_softmax` involved -- that's called
    /// only by `Loss::loss`, above this function), so this is an exact check, not just a
    /// finite/non-zero sanity check.
    #[test]
    fn mdd_wmc_loss_gradient_matches_finite_differences_via_autodiff() {
        use burn::backend::Autodiff;

        type ADBackend = Autodiff<NdArray>;
        let ad_device = NdArrayDevice::default();
        let plain_device = NdArrayDevice::default();
        let domain_size = 3;

        let (batch_ad, loss_ad) = two_sample_batch::<ADBackend>(&ad_device, 0.0);
        let (batch_plain, loss_plain) = two_sample_batch::<NdArray>(&plain_device, 0.0);

        let flat: Vec<f32> = vec![
            0.2, 0.5, 0.3, 0.1, 0.3, 0.6, 0.4, 0.4, 0.2, 0.6, 0.3, 0.1, 0.2, 0.2, 0.6, 0.5, 0.25,
            0.25,
        ];

        let probs_ad: Tensor<ADBackend, 3> =
            Tensor::<ADBackend, 1>::from_data(flat.as_slice(), &ad_device)
                .reshape([2, 3, domain_size])
                .require_grad();
        let loss = loss_ad.wmc_loss(probs_ad.clone(), &batch_ad);
        let grads = loss.backward();
        let grad = probs_ad
            .grad(&grads)
            .expect("probs should have a gradient after backward()");
        let grad_values: Vec<f32> = grad.into_data().to_vec::<f32>().unwrap();

        let base_tensor: Tensor<NdArray, 3> =
            Tensor::<NdArray, 1>::from_data(flat.as_slice(), &plain_device).reshape([
                2,
                3,
                domain_size,
            ]);
        let base_loss_tensor = loss_plain.wmc_loss(base_tensor, &batch_plain);
        let base_loss: f64 = base_loss_tensor.into_data().to_vec::<f32>().unwrap()[0] as f64;

        let eps = 1e-4f32;
        for i in 0..flat.len() {
            let mut bumped = flat.clone();
            bumped[i] += eps;
            let bumped_tensor: Tensor<NdArray, 3> =
                Tensor::<NdArray, 1>::from_data(bumped.as_slice(), &plain_device).reshape([
                    2,
                    3,
                    domain_size,
                ]);
            let bumped_loss_tensor = loss_plain.wmc_loss(bumped_tensor, &batch_plain);
            let bumped_loss: f64 =
                bumped_loss_tensor.into_data().to_vec::<f32>().unwrap()[0] as f64;
            let finite_diff = (bumped_loss - base_loss) / eps as f64;
            assert!(
                (finite_diff - grad_values[i] as f64).abs() < 1e-2,
                "index {i}: analytic={} finite_diff={finite_diff}",
                grad_values[i],
            );
        }
    }

    /// Without noise (`gumbel_scale` zero) the loss is a deterministic function of the logits, and
    /// with noise (the default) two evaluations of the same logits differ.
    #[test]
    fn gumbel_scale_zero_makes_the_loss_deterministic() {
        use burn::tensor::Distribution;

        type Plain = NdArray;
        let device = NdArrayDevice::default();
        let (batch, loss_fn) = two_sample_batch::<Plain>(&device, 1.0);
        let logits: Tensor<Plain, 3> =
            Tensor::random([2, 3, 3], Distribution::Uniform(-1.0, 1.0), &device);
        let value = |loss_fn: &DecomposedMddLoss| -> f32 {
            loss_fn
                .loss(logits.clone(), &batch)
                .into_data()
                .to_vec::<f32>()
                .unwrap()[0]
        };

        let noisy = value(&loss_fn);
        assert_ne!(
            noisy,
            value(&loss_fn),
            "the default scale draws fresh noise"
        );

        let loss_fn = loss_fn.with_gumbel_scale(0.0);
        let first = value(&loss_fn);
        assert!(first.is_finite());
        assert_eq!(first, value(&loss_fn), "scale zero draws no noise");
    }

    /// The loss of fixed logits on a batch of the two problems of `two_sample_batch`, without
    /// noise.
    fn noiseless_loss(bp_rounds: usize, mask_fraction: f64, skip_infeasible: bool) -> f32 {
        let device = NdArrayDevice::default();
        let (batch, _) = two_sample_batch::<NdArray>(&device, mask_fraction);
        let logits: Tensor<NdArray, 3> = Tensor::from_data(
            [
                [[0.3, -0.2, 0.9], [1.1, 0.0, -0.5], [-0.7, 0.4, 0.2]],
                [[0.0, 0.8, -0.3], [0.5, 0.5, -1.0], [0.2, -0.6, 0.9]],
            ],
            &device,
        );
        let mut loss_fn = DecomposedMddLoss::new(MddCompilationConfig::default(), 3)
            .with_bp_rounds(bp_rounds)
            .with_gumbel_scale(0.0)
            .with_infeasible_constraints_skipped(skip_infeasible);
        loss_fn.compile(&batch.problems);
        loss_fn
            .loss(logits, &batch)
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0]
    }

    /// With every variable re-predicted every constraint of these problems is satisfiable, so
    /// skipping the infeasible ones changes nothing, with or without belief propagation.
    #[test]
    fn skipping_infeasible_constraints_changes_nothing_when_every_variable_is_re_predicted() {
        for rounds in [0, 2] {
            let all = noiseless_loss(rounds, 1.0, false);
            let skipped = noiseless_loss(rounds, 1.0, true);
            assert!(all > 0.0 && all.is_finite());
            assert_eq!(all, skipped, "bp_rounds={rounds}");
        }
    }

    /// With every variable kept, a constraint is satisfied (a count of one) or infeasible. Skipping
    /// the infeasible ones leaves a loss of about zero, while counting them adds
    /// `-ln(WMC_EPS)` for each.
    #[test]
    fn skipped_constraints_do_not_add_the_loss_of_an_infeasible_count() {
        for rounds in [0, 2] {
            let skipped = noiseless_loss(rounds, 0.0, true);
            let all = noiseless_loss(rounds, 0.0, false);
            assert!(skipped.abs() < 1e-5, "bp_rounds={rounds}: {skipped}");
            assert!(all >= skipped, "bp_rounds={rounds}: {all} < {skipped}");
        }
    }

    /// The loss counts the rounds of belief propagation of every sample until they are taken.
    #[test]
    fn the_rounds_of_belief_propagation_are_counted_per_sample_until_taken() {
        let device = NdArrayDevice::default();
        let (batch, loss_fn) = two_sample_batch::<NdArray>(&device, 1.0);
        let logits: Tensor<NdArray, 3> = Tensor::random(
            [2, 3, 3],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let plain = <DecomposedMddLoss as Loss<NdArray, _>>::take_bp_rounds(&loss_fn);
        assert_eq!(plain.count(), 0);

        let loss_fn = loss_fn.with_bp_rounds(3).with_gumbel_scale(0.0);
        loss_fn.loss(logits.clone(), &batch);
        loss_fn.loss(logits, &batch);
        let rounds = <DecomposedMddLoss as Loss<NdArray, _>>::take_bp_rounds(&loss_fn);
        assert_eq!(rounds.count(), 4, "two samples, twice");
        assert_eq!(rounds.mean(), Some(3.0));
        let again = <DecomposedMddLoss as Loss<NdArray, _>>::take_bp_rounds(&loss_fn);
        assert_eq!(again.count(), 0, "taking them resets the count");
    }

    #[test]
    #[should_panic(expected = "Gumbel scale")]
    fn a_negative_gumbel_scale_is_refused() {
        let _ = DecomposedMddLoss::new(MddCompilationConfig::default(), 3).with_gumbel_scale(-1.0);
    }

    /// End-to-end through the full `Loss::loss` path (`gumbel_softmax` + `blend_with_current` +
    /// `mdd_wmc_loss`): with `logits` requiring grad, `.backward()` should reach `logits` with a
    /// finite, non-all-zero gradient of the expected shape. Unlike the test above, this can't be
    /// checked against an exact finite-difference value -- `gumbel_softmax` draws fresh noise on
    /// every call -- so it's a sanity check that the gradient actually flows end to end, not a
    /// correctness check on the custom gradient math itself (that's covered above).
    #[test]
    fn loss_backpropagates_through_gumbel_and_blend() {
        use burn::backend::Autodiff;
        use burn::tensor::Distribution;

        type ADBackend = Autodiff<NdArray>;

        let device = NdArrayDevice::default();
        let (batch, loss_fn) = two_sample_batch::<ADBackend>(&device, 1.0);

        let logits: Tensor<ADBackend, 3> =
            Tensor::random([2, 3, 3], Distribution::Uniform(-1.0, 1.0), &device).require_grad();

        let loss = loss_fn.loss(logits.clone(), &batch);
        let loss_value: f32 = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
        assert!(
            loss_value.is_finite(),
            "loss should be finite, got {loss_value}"
        );

        let grads = loss.backward();
        let grad = logits
            .grad(&grads)
            .expect("logits should have a gradient after backward()");
        assert_eq!(grad.dims(), [2, 3, 3]);

        let grad_values: Vec<f32> = grad.into_data().to_vec::<f32>().unwrap();
        assert!(
            grad_values.iter().all(|v| v.is_finite()),
            "every gradient entry should be finite"
        );
        assert!(
            grad_values.iter().any(|&v| v != 0.0),
            "gradient should not be identically zero"
        );
    }

    /// A `NotEquals` constraint both of whose variables are pinned (`var_masks` all false) to the
    /// *same* value is unconditionally violated: `wmc = 0` for that MDD regardless of what the
    /// network predicts, so `mdd_loss_and_gradient`'s `-log(wmc+eps)`/`-1/(wmc+eps)` genuinely hit
    /// their `WMC_EPS` floor rather than some comfortably-nonzero value. This checks the loss and
    /// its gradient both stay finite in that regime -- the case `WMC_EPS` exists for.
    #[test]
    fn loss_stays_finite_when_a_pinned_assignment_violates_a_constraint() {
        use burn::backend::Autodiff;

        type ADBackend = Autodiff<NdArray>;
        let device = NdArrayDevice::default();

        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2]);
        let y = problem.add_variable(vec![0, 1, 2]);
        not_equals(&mut problem, x, y);
        let problem = Arc::new(problem);

        let mut loss_fn = DecomposedMddLoss::new(MddCompilationConfig::default(), 3);
        loss_fn.compile(&[Arc::clone(&problem)]);
        let dataset = ConsFormerDataset::<ADBackend>::new(vec![problem], &device);
        let samples: Vec<_> = (0..dataset.len())
            .map(|i| dataset.get(i).unwrap())
            .collect();
        let batcher = ConsFormerBatcher {
            mask_fraction: 0.0,
            destroy: None,
        };
        let batch = batcher.batch(samples, &device);

        // `mask_fraction = 0.0` above means `var_masks` is all-false, so `blend_with_current`
        // pins every variable to `batch.assignments` regardless of `logits` -- setting both
        // assignments to 0 violates `not_equals(x, y)` unconditionally.
        let assignments: Tensor<ADBackend, 2, Int> =
            Tensor::<ADBackend, 1, Int>::from_data([0i64, 0i64].as_slice(), &device)
                .reshape([1, 2]);
        let batch = ConsFormerBatch {
            assignments,
            ..batch
        };

        let logits: Tensor<ADBackend, 3> = Tensor::zeros([1, 2, 3], &device).require_grad();

        let loss = loss_fn.loss(logits.clone(), &batch);
        let loss_value: f32 = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
        assert!(
            loss_value.is_finite(),
            "loss should be finite, got {loss_value}"
        );
        assert!(
            (loss_value as f64 - (-(WMC_EPS).ln())).abs() < 1e-3,
            "an unconditionally-violated constraint should read back as -log(WMC_EPS), got {loss_value}",
        );

        let grads = loss.backward();
        let grad = logits
            .grad(&grads)
            .expect("logits should have a gradient after backward()");
        let grad_values: Vec<f32> = grad.into_data().to_vec::<f32>().unwrap();
        assert!(
            grad_values.iter().all(|v| v.is_finite()),
            "every gradient entry should be finite even when a constraint is unconditionally violated -- got {:?}",
            grad_values,
        );
    }

    /// Direct correctness check for the weight-masking fix `layer_weights_from_probs` needs now
    /// that `constraint.structure` is compiled with no per-instance domain restriction (see
    /// `crate::mdd::arena`'s module doc): a real "given" position (its domain narrowed to a
    /// single value, e.g. a Sudoku clue) must have every *other* raw value's weight forced to
    /// zero, even though the shared, un-hinted `AllDifferent` structure has a real edge for each
    /// of them. Checks both that the masked weight vector itself is a one-hot at the given value,
    /// and that `wmc` over those masked weights exactly matches a brute-force count restricted to
    /// assignments honouring the given -- i.e. sampling/`wmc`/gradient never actually credits an
    /// assignment that uses a masked-out value at the given position.
    #[test]
    fn layer_weights_from_probs_masks_out_a_given_values_forbidden_domain() {
        let domain_size = 4;
        let mut problem = Problem::default();
        let vars = problem.add_variables(4, (0..domain_size as isize).collect());
        all_different(&mut problem, vars.clone());
        // `vars[0]` is a "given": its real domain is narrowed to a single value, but the shared
        // template below is still compiled over the full nominal domain (no hint baked in).
        crate::modelling::equal(&mut problem, vars[0], 2);
        let problem = Arc::new(problem);

        let arena = MddArena::new();
        let constraint = problem
            .iter_constraints()
            .next()
            .expect("problem must have exactly one constraint");
        let constraint = crate::mdd::arena::compile_constraint(
            &arena,
            &problem,
            constraint,
            &crate::mdd::heuristics::OrderingHeuristic::MinDomMaxLinked,
            &crate::mdd::arena::nominal_domain(domain_size),
            usize::MAX,
            usize::MAX,
        );

        // Uniform raw probabilities everywhere -- if masking didn't happen, every position
        // (including the given's 3 forbidden values) would carry equal, nonzero weight.
        let probs_for_sample: Vec<f32> = vec![0.25; 4 * domain_size];
        let weights =
            layer_weights_from_probs(&constraint, &problem, &probs_for_sample, domain_size);

        let given_layer = (0..constraint.order.len())
            .find(|&l| constraint.decision_at_layer(l) == vars[0])
            .expect("vars[0] must be in the compiled constraint's scope");
        assert_eq!(
            weights[given_layer],
            vec![0.0, 0.0, 0.25, 0.0],
            "a given position's weight vector must be zero everywhere but its one allowed value"
        );

        let wmc = crate::mdd::wmc::wmc(&constraint, &weights);

        // Brute force: every permutation of {0,1,2,3} over the 4 variables with vars[0] fixed to
        // 2, each weighted 0.25^4 (uniform raw probability at every position).
        let mut brute = 0.0;
        for a in 0..domain_size {
            for b in 0..domain_size {
                for c in 0..domain_size {
                    let full = [2usize, a, b, c];
                    let vals: std::collections::HashSet<usize> = full.iter().copied().collect();
                    if vals.len() == 4 {
                        brute += 0.25f64.powi(4);
                    }
                }
            }
        }
        assert!(
            (wmc - brute).abs() < 1e-9,
            "wmc={wmc} brute={brute} -- masked wmc must match assignments honouring the given"
        );
    }
    /// A problem with an `AllDifferent` over three variables and a `NotEquals` over the first two,
    /// compiled.
    fn two_constraint_problem() -> (Arc<Problem>, DecomposedMddLoss) {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2]);
        all_different(&mut problem, vars.clone());
        not_equals(&mut problem, vars[0], vars[1]);
        let problem = Arc::new(problem);
        let mut loss_fn = DecomposedMddLoss::new(MddCompilationConfig::default(), 3);
        loss_fn.compile(&[Arc::clone(&problem)]);
        (problem, loss_fn)
    }

    /// A constraint can be completed when some solution agrees with its kept values: keeping
    /// `x2` alone leaves both constraints of the problem satisfiable, keeping `x0 = x1` neither.
    #[test]
    fn completable_constraints_have_a_solution_that_agrees_with_the_kept_values() {
        let (problem, loss_fn) = two_constraint_problem();
        let mdds = loss_fn.compiled_for(&problem);
        let keeping = |kept: &[(usize, usize)]| Evidence {
            assignment: (0..3)
                .map(|v| ValueIndex(kept.iter().find(|k| k.0 == v).map_or(0, |k| k.1)))
                .collect(),
            decided: (0..3).map(|v| kept.iter().any(|k| k.0 == v)).collect(),
        };
        let completable_count = |evidence: &Evidence| {
            mdds.iter()
                .filter(|c| completable(c, &problem, evidence, 3))
                .count()
        };
        assert_eq!(completable_count(&Evidence::free(3)), 2);
        assert_eq!(completable_count(&keeping(&[(2, 1)])), 2);
        assert_eq!(completable_count(&keeping(&[(0, 1), (1, 1)])), 0);
        assert_eq!(completable_count(&keeping(&[(0, 0), (1, 1)])), 2);
    }

    const TWO_CONSTRAINT_PROBS: [f32; 9] = [0.2, 0.5, 0.3, 0.1, 0.3, 0.6, 0.4, 0.4, 0.2];

    #[test]
    fn belief_propagation_changes_the_loss_of_several_constraints() {
        let (problem, loss_fn) = two_constraint_problem();
        let mdds = loss_fn.compiled_for(&problem);
        let free = Evidence::free(3);
        let (plain, _) =
            sample_loss_and_gradient(mdds, &problem, &TWO_CONSTRAINT_PROBS, 3, 3, 0, &free, false);
        let (refined, grad) =
            sample_loss_and_gradient(mdds, &problem, &TWO_CONSTRAINT_PROBS, 3, 3, 1, &free, false);
        assert!(refined.is_finite() && grad.iter().all(|g| g.is_finite()));
        assert!(
            (plain - refined).abs() > 1e-6,
            "the other constraints should change the labels: {plain} vs {refined}"
        );
    }

    #[test]
    fn belief_propagation_changes_nothing_for_a_single_constraint() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2]);
        all_different(&mut problem, vars);
        let problem = Arc::new(problem);
        let mut loss_fn = DecomposedMddLoss::new(MddCompilationConfig::default(), 3);
        loss_fn.compile(&[Arc::clone(&problem)]);
        let mdds = loss_fn.compiled_for(&problem);
        let free = Evidence::free(3);

        let (plain, plain_grad) =
            sample_loss_and_gradient(mdds, &problem, &TWO_CONSTRAINT_PROBS, 3, 3, 0, &free, false);
        for rounds in [1, 5] {
            let (refined, grad) = sample_loss_and_gradient(
                mdds,
                &problem,
                &TWO_CONSTRAINT_PROBS,
                3,
                3,
                rounds,
                &free,
                false,
            );
            assert!((plain - refined).abs() < 1e-9, "{plain} vs {refined}");
            for (a, b) in plain_grad.iter().zip(&grad) {
                assert!((a - b).abs() < 1e-5, "{plain_grad:?} vs {grad:?}");
            }
        }
    }

    /// The gradient treats the messages of belief propagation as constants, so it is checked
    /// against finite differences of the loss with the cavity factors held fixed.
    #[test]
    fn gradient_through_the_tilt_matches_finite_differences_with_fixed_messages() {
        let (problem, loss_fn) = two_constraint_problem();
        let mdds = loss_fn.compiled_for(&problem);
        let free = Evidence::free(3);
        let factors =
            sample_cavity_factors(mdds, &problem, &TWO_CONSTRAINT_PROBS, 3, 3, 2, &free).0;

        let total = |probs: &[f32]| -> (f64, Vec<f32>) {
            let mut grad = vec![0.0f32; 9];
            let mut loss = 0.0;
            for (index, constraint) in mdds.iter().enumerate() {
                loss += mdd_loss_and_gradient(
                    constraint,
                    &problem,
                    probs,
                    3,
                    Some(factors[index].as_slice()),
                    &mut grad,
                );
            }
            (loss, grad)
        };
        let (base, grad) = total(&TWO_CONSTRAINT_PROBS);
        let eps = 1e-3f32;
        for i in 0..9 {
            let mut bumped = TWO_CONSTRAINT_PROBS;
            bumped[i] += eps;
            let finite_difference = (total(&bumped).0 - base) / eps as f64;
            assert!(
                (finite_difference - grad[i] as f64).abs() < 1e-2,
                "index {i}: analytic={} finite_diff={finite_difference}",
                grad[i],
            );
        }
    }

    /// The gradient through the tilt as it was first written, in terms of the labels: with
    /// $a = q \odot f$, $A = \sum a$ and $S = \sum q$ the labels are $S a / A$ and
    /// $\partial L / \partial q_u = T + S f_u (g_u - T) / A$, $T = \sum_v g_v a_v / A$.
    /// Kept as the reference that `prior_gradient` must reproduce.
    fn tilt_backward_reference(weights: &[f64], factor: &[f64], grad: &[f64]) -> Vec<f64> {
        let factor = floored(factor);
        let mass: f64 = weights.iter().sum();
        let tilted: Vec<f64> = weights.iter().zip(&factor).map(|(w, f)| w * f).collect();
        let total: f64 = tilted.iter().sum();
        if !(total > 0.0 && total.is_finite()) {
            return grad.to_vec();
        }
        let mean: f64 = grad.iter().zip(&tilted).map(|(g, a)| g * a).sum::<f64>() / total;
        grad.iter()
            .zip(&factor)
            .map(|(g, f)| mean + mass * f * (g - mean) / total)
            .collect()
    }

    /// Writing the gradient with the messages of belief propagation changes nothing: it equals the
    /// first formulation on the gradients of real diagrams, for ordinary and extreme
    /// probabilities and for priors that do not sum to one.
    #[test]
    fn the_gradient_in_belief_propagation_terms_equals_the_tilt_formula() {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};

        let (problem, loss_fn) = two_constraint_problem();
        let mdds = loss_fn.compiled_for(&problem);
        let mut rng = StdRng::seed_from_u64(11);
        let mut compared = 0;
        for trial in 0..300 {
            let spread = if trial % 2 == 0 { 2.0 } else { 20.0 };
            // With near one-hot probabilities the cavity factors are floored at 1e-12 and the
            // two formulations, which cancel large terms differently, agree to fewer digits.
            let tolerance = if trial % 2 == 0 { 1e-9 } else { 1e-3 };
            let mut probs = Vec::new();
            for _ in 0..3 {
                let logits: Vec<f64> = (0..3).map(|_| rng.random_range(-spread..spread)).collect();
                let largest = logits.iter().copied().fold(f64::MIN, f64::max);
                let weights: Vec<f64> = logits.iter().map(|l| (l - largest).exp()).collect();
                let total: f64 = weights.iter().sum();
                probs.extend(weights.iter().map(|w| (w / total) as f32));
            }
            let free = Evidence::free(3);
            let rounds = 1 + trial % 3;
            let factors = sample_cavity_factors(mdds, &problem, &probs, 3, 3, rounds, &free).0;
            for (index, constraint) in mdds.iter().enumerate() {
                let mut prior = layer_weights_from_probs(constraint, &problem, &probs, 3);
                if trial % 3 == 0 {
                    // Probability outside the domain: the prior sums to less than one.
                    let kept = rng.random_range(0.3..1.0);
                    prior.iter_mut().flatten().for_each(|w| *w *= kept);
                }
                let labels: Vec<Vec<f64>> = prior
                    .iter()
                    .zip(&factors[index])
                    .map(|(w, f)| cavity_labels(w, f))
                    .collect();
                let (log_wmc, gradient) = log_wmc_and_gradient(constraint, &labels);
                let log_total = log_add_exp(log_wmc, WMC_EPS.ln());
                let chain = -(log_wmc - log_total).exp();
                for layer in 0..prior.len() {
                    let scaled: Vec<f64> = gradient[layer].iter().map(|g| chain * g).collect();
                    let reference =
                        tilt_backward_reference(&prior[layer], &factors[index][layer], &scaled);
                    let ours = prior_gradient(
                        &prior[layer],
                        &factors[index][layer],
                        &gradient[layer],
                        chain,
                    );
                    for (a, b) in reference.iter().zip(&ours) {
                        let scale = 1.0 + a.abs().max(b.abs());
                        let relative = (a - b).abs() / scale;
                        assert!(
                            relative < tolerance,
                            "trial {trial}, layer {layer}: {reference:?} vs {ours:?}"
                        );
                        compared += 1;
                    }
                }
            }
        }
        assert!(compared > 1000);
    }

    #[test]
    fn wmc_loss_with_belief_propagation_is_finite_and_differs_from_without() {
        let device = NdArrayDevice::default();
        let (batch, loss_fn) = two_sample_batch::<NdArray>(&device, 1.0);
        let flat: Vec<f32> = [0.2, 0.5, 0.3, 0.1, 0.3, 0.6, 0.4, 0.4, 0.2]
            .iter()
            .chain(&[0.6, 0.3, 0.1, 0.2, 0.2, 0.6, 0.5, 0.25, 0.25])
            .copied()
            .collect();
        let probs = || -> Tensor<NdArray, 3> {
            Tensor::<NdArray, 1>::from_data(flat.as_slice(), &device).reshape([2, 3, 3])
        };
        let value = |loss: Tensor<NdArray, 1>| loss.into_data().to_vec::<f32>().unwrap()[0];

        let plain = value(loss_fn.wmc_loss(probs(), &batch));
        let refined = value(loss_fn.with_bp_rounds(3).wmc_loss(probs(), &batch));
        assert!(refined.is_finite());
        assert!((plain - refined).abs() > 1e-6, "{plain} vs {refined}");
    }

    /// Confident networks give near one-hot probabilities that the other constraints rule out,
    /// which must neither collapse the weights of a diagram nor overflow its gradient.
    #[test]
    fn belief_propagation_keeps_the_loss_and_gradient_finite_for_extreme_probabilities() {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};

        let (problem, loss_fn) = two_constraint_problem();
        let mdds = loss_fn.compiled_for(&problem);
        let mut rng = StdRng::seed_from_u64(7);
        for trial in 0..200 {
            let mut probs = Vec::new();
            for _ in 0..3 {
                // Logits spread over up to 40 nats: probabilities down to 1e-17 and exact zeros
                // once rounded to f32.
                let logits: Vec<f64> = (0..3).map(|_| rng.random_range(-20.0..20.0)).collect();
                let largest = logits.iter().copied().fold(f64::MIN, f64::max);
                let weights: Vec<f64> = logits.iter().map(|l| (l - largest).exp()).collect();
                let total: f64 = weights.iter().sum();
                probs.extend(weights.iter().map(|w| (w / total) as f32));
            }
            for rounds in [1, 3, 5] {
                let (loss, grad) = sample_loss_and_gradient(
                    mdds,
                    &problem,
                    &probs,
                    3,
                    3,
                    rounds,
                    &Evidence::free(3),
                    false,
                );
                assert!(
                    loss.is_finite() && grad.iter().all(|g| g.is_finite()),
                    "trial {trial}, {rounds} rounds: probs {probs:?} gave {loss} and {grad:?}"
                );
            }
        }
    }

    #[test]
    fn log_add_exp_matches_the_direct_sum_and_survives_extremes() {
        assert!((log_add_exp(0.3f64.ln(), 0.2f64.ln()) - 0.5f64.ln()).abs() < 1e-12);
        // A count of exp(-2000) next to the epsilon: the direct sum is just the epsilon.
        assert!((log_add_exp(-2000.0, 1e-6f64.ln()) - 1e-6f64.ln()).abs() < 1e-12);
        assert_eq!(log_add_exp(f64::NEG_INFINITY, 1e-6f64.ln()), 1e-6f64.ln());
    }
}
