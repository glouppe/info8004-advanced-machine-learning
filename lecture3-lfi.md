class: middle, center, title-slide
count: false

# Advanced Machine Learning

Neural Likelihood-free Inference

<br><br>
Prof. Gilles Louppe<br>
[g.louppe@uliege.be](g.louppe@uliege.be)

---

class: middle, black-slide

.center[<video controls autoplay loop muted preload="auto" height="480" width="640">
  <source src="./figures/lec3/galton.mp4" type="video/mp4">
</video>]

---

<br>
.center.width-70[![](./figures/lec3/paths.png)]

How to estimate the probability $\theta$ of going left when hitting a pin?

???

Intermediate question to answer: what is the probability of observing a ball in a particular bin?
- Take the leftmost path first
- Take the next bin
- Take the middle bin

---

count: false

<br>
.center.width-70[![](./figures/lec3/paths.png)]

The probability of ending in bin $x$ corresponds to the total probability of all the paths $z$ from start to $x$,
$$\begin{aligned}
p(x | \theta)
= \int p(x,z|\theta) dz
= \begin{pmatrix}
n \\\\
x
\end{pmatrix}
\theta^x (1-\theta)^{n-x}.
\end{aligned}$$

Therefore $\hat{\theta} = \arg \max \prod\_{x\_i} p(x\_i|\theta) \pi(\theta)$.

???

The likelihood function $p(x|\theta)$ enables inference:
- Maximum likelihood estimation: $\theta^* = \arg \max\_\theta p(x|\theta)$
- Bayesian posterior inference: $p(\theta|x) = p(x|\theta) p(\theta) / p(x)$

---

count: false

<br>
.center.width-70[![](./figures/lec3/paths.png)]

But what if we shift or remove some of the pins?

---

class: middle

The Galton board is a *metaphore* of simulation-based science:
.grid.center[
.kol-2-5[Galton board device]
.kol-1-5[$\to$]
.kol-2-5[Computer simulation]
]
.grid.center[
.kol-2-5[Parameters $\theta$]
.kol-1-5[$\to$]
.kol-2-5[Model parameters $\theta$]
]
.grid.center[
.kol-2-5[Buckets $x$]
.kol-1-5[$\to$]
.kol-2-5[Observables $x$]
]
.grid.center[
.kol-2-5[Random paths $z$]
.kol-1-5[$\to$]
.kol-2-5[Latent variables $z$<br> (stochastic execution traces through simulator)]
]

Inference in this context requires **likelihood-free algorithms**.

---

class: middle

.center.width-100[![](./figures/lec3/lfi-setup2.png)]

.footnote[Credits: Johann Brehmer.]

???

Many fields of science have developed mechanistic, stochastic, models
that can be used to predict how a system will behave in a variety of circumstances.


---

# A thriving field of research

.grid[
.kol-1-2.center[
.center.width-80[![](./figures/lec3/pp.png)]
Particle physics
]
.kol-1-2.center[
.center.width-40[![](./figures/lec3/proteins.png)]
Protein folding
]
]
.grid[
.kol-1-2.center[
.center.width-60[![](./figures/lec3/contagion.jpg)]
Epidemiology
]
.kol-1-2.center[
.center.width-60[![](./figures/lec3/climate-nasa.gif)]
Climate science
]
]
.grid[
.kol-1-2.center[
.grid[
.kol-1-3.width-100[![](figures/lec3/planck.png)]
.kol-1-3.width-100[![](figures/lec3/illustris1.png)]
.kol-1-3.width-70[![](figures/lec3/illustris2.png)]
]
Astrophysics and cosmology
]
.kol-1-2.center[<br><br>(... and many others!)]
]

???

Scientific simulators span the entire range of distance scales with notable examples including:
- particle physics,
- molecular dynamics,
- protein folding,
- population genetics,
- neuroscience,
- epidemiology,
- economics,
- ecology,
- climate science,
- astrophysics,
- cosmology

---

class: center, middle, black-slide

.width-70[![](figures/lec3/sm.jpg)]

---

background-image: url(./figures/lec3/lhc.gif)
class: center, middle, black-slide

---

background-image: url(./figures/lec3/englert.jpg)
class: center, middle, black-slide

---

class: middle

## Particle physics

.center.width-90[![](./figures/lec3/lfi-chain.png)]
.grid[
.kol-1-5.center[
SM with parameters $\theta$

.width-100[![](./figures/lec3/sm.png)]]
.kol-2-5.center[
Simulated observables $x$

.width-80[![](./figures/lec3/lhc.gif)]]
.kol-2-5.center[
Real observations $x\_\text{obs}$

.width-80[![](./figures/lec3/pp-xobs1.png)]
.width-80[![](./figures/lec3/pp-xobs2.jpeg)]]
]

---

class: middle

.width-100[![](figures/lec3/process1.png)]

.footnote[Credits: Johann Brehmer.]

???

generation: pencil and paper calculable from first principles

---

count: false
class: middle

.width-100[![](figures/lec3/process2.png)]

.footnote[Credits: Johann Brehmer.]

???

parton shower + hadronization: controlled approximation of first principles + phenomenological model

---

count: false
class: middle

.width-100[![](figures/lec3/process3.png)]

.footnote[Credits: Johann Brehmer.]

???

detector simulation: interaction with the materials and digitization

---

count: false
class: middle

.width-100[![](figures/lec3/process4.png)]

.footnote[Credits: Johann Brehmer.]

???

reconstruction simulation

---

class: middle

$$p(x|\theta) = \underbrace{\iiint}\_{\text{intractable}} p(z\_p|\theta) p(z\_s|z\_p) p(z\_d|z\_s) p(x|z\_d) dz\_p dz\_s dz\_d$$

???

That's bad!

---

# Ingredients

Statistical inference requires the computation of *key ingredients*, such as
- the likelihood $p(x|\theta)$,
- the likelihood ratio $r(x|\theta\_0,\theta\_1) = \frac{p(x|\theta\_0)}{p(x|\theta\_1)}$,
- or the posterior $p(\theta|x)$.

In the simulator-based scenario, each of these ingredients can be approximated with modern machine learning techniques, **even if none are tractable during training**!

---

# Likelihood ratio

The likelihood ratio
$$r(x|\theta\_0,\theta\_1) = \frac{p(x|\theta\_0)}{p(x|\theta\_1)}$$
is the quantity that is *central* to many **statistical inference** procedures.

## Examples

- Frequentist hypothesis testing
- Bayesian model comparison
- Bayesian posterior inference with MCMC or VI
- Supervised learning
- Generative adversarial networks
- Empirical Bayes with Adversarial Variational Optimization
- Optimal compression

???

... but not too bad.

Point = it is ok if the likelihood cannot be evaluated, because the quantity that is sufficient for most statistical inference procedures is actually the likelihood ratio.

---

class: middle
count: false

# Frequentist inference

---

# The frequentist (physicist's) way

.grid[
.kol-3-4[
The Neyman-Pearson lemma states that the likelihood ratio
$$r(x|\theta\_0,\theta\_1) = \frac{p(x|\theta\_0)}{p(x|\theta\_1)}$$
is the **most powerful test statistic** to discriminate between a null hypothesis $\theta\_0$ and an alternative $\theta\_1$.
]
.kol-1-4[<br>.width-100[![](figures/lec3/ellipse.png)]]
]

.center.width-70[![](figures/lec3/lemma.png)]

???

- The first, most obvious example that illustrates the importance of the likelihood ratio is the Neyman-Pearson lemma, which ...

- The LR test assesses the goodness of fit of two competing hypothesis. It applies to simple and composite hypotheses.

- ... but how do you compute this ratio in the likelihood-free scenario?

---

class: middle

.center.width-90[![](./figures/lec3/lfi-summary-stats.png)]

Define a projection function $s:\mathcal{X} \to \mathbb{R}$ mapping observables $x$ to a summary statistic $x'=s(x)$.

Then, **approximate** the likelihood $p(x|\theta)$ with the surrogate $\hat{p}(x|\theta) = p(x'|\theta)$.

From this it comes
$$\frac{p(x|\theta\_0)}{p(x|\theta\_1)} \approx \frac{\hat{p}(x|\theta\_0)}{\hat{p}(x|\theta\_1)} = \hat{r}(x|\theta\_0,\theta\_1).$$

---

class: middle

## Wilks theorem

Consider the test statistic $$q(\theta) = -2 \sum\_x \log \frac{p(x|\theta)}{p(x|\hat{\theta})} = -2 \sum\_x \log r(x|\theta,\hat{\theta})$$ for a fixed number $N$ of observations $\\{x\\}$ and where $\hat{\theta}$ is the maximum likelihood estimator.

When $N \to \infty$, $q(\theta) \sim \chi\_2$.

Therefore (and provided the assumptions apply!), an observed value $q\_\text{obs}(\theta)$ translates directly to a p-value that measures the confidence with which $\theta$ can be excluded.

---

class: middle, center

.grid[
.kol-2-3[.width-100[![](figures/lec3/higgs4l.gif)]]
.kol-1-3[<br>.width-100[![](figures/lec3/higgs-discovery.jpg)]

.width-100[![](figures/lec3/nobel.jpg)]]
]


Discovery of the Higgs boson at 5-$\sigma$

???

m4l = four-lepton invariant mass

Q: How to choose $s$?

---

# Cᴀʀʟ

Supervised learning provides a way to **automatically** construct $s$:
- Let us consider a neural network classifier $\hat{s}$ tasked to distinguish $x\_i \sim p(x|\theta\_0)$ labelled $y\_i=0$ from $x\_i \sim p(x|\theta\_1)$ labelled $y\_i=1$.
- Train $\hat{s}$ by minimizing the cross-entropy loss.

.center.width-50[![](figures/lec3/s_x.png)]

.footnote[Cranmer, Pavez and Louppe, 2015 [[arXiv:1506.02169](https://arxiv.org/abs/1506.02169)].]

???

Explain the figure

---

class: middle

The solution $\hat{s}$ found after training  approximates the optimal classifier
$$\hat{s}(x) \approx s^\*(x) = \frac{p(x|\theta\_1)}{p(x|\theta\_0)+p(x|\theta\_1)}.$$

Therefore, $$r(x|\theta\_0,\theta\_1) \approx \hat{r}(x|\theta\_0,\theta\_1)=\frac{1-\hat{s}(x)}{\hat{s}(x)}$$

That is, **supervised classification** is equivalent to *likelihood ratio estimation*.

.footnote[Cranmer, Pavez and Louppe, 2015 [[arXiv:1506.02169](https://arxiv.org/abs/1506.02169)].]

---

class: middle

.center.width-90[![](figures/lec3/inference-1.png)]

To avoid retraining a classifier $\hat{s}$ for every $(\theta\_0, \theta\_1)$ pair, fix $\theta\_1$ to $\theta\_\text{ref}$ and train a single **parameterized** classifier $\hat{s}(x|\theta\_0,\theta\_\text{ref})$ where $\theta\_0$ is also given as input.

Therefore, we have
$$\hat{r}(x|\theta\_0,\theta\_\text{ref}) = \frac{1 - \hat{s}(x|\theta\_0,\theta\_\text{ref})}{\hat{s}(x|\theta\_0,\theta\_\text{ref})}$$
such that for any $(\theta\_0, \theta\_1)$,
$$r(x|\theta\_0,\theta\_1) \approx \frac{\hat{r}(x|\theta\_0,\theta\_\text{ref})}{\hat{r}(x|\theta\_1,\theta\_\text{ref})}.$$

.footnote[Cranmer, Pavez and Louppe, 2015 [[arXiv:1506.02169](https://arxiv.org/abs/1506.02169)].]

---

# Opening the black box

<br><br><br>
.center[.width-30[![](figures/lec3/blackbox.png)] &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; .width-30[![](figures/lec3/blackbox2.png)]]

.grid[
.kol-1-2[Traditional likelihood-free inference treats the simulator as a generative **black box**: parameters in, samples out.]
.kol-1-2[But in most real-life problems, we have access to the simulator code and some understanding of the microscopic processes.]
]

---

class: middle

.center[.width-50[![](./figures/lec3/paths.png)]

$p(x|\theta)$ is usually intractable. What about $p(x,z|\theta)$?]

---

class: middle

## Extracting the joint likelihood ratio

.width-60.center[![](figures/lec3/mining.png)]

For each run, we can calculate the probability of the chosen path for different values of the parameters and the joint likelihood-ratio:

$$r(x,z|\theta\_0, \theta\_1) = \frac{p(x,z|\theta\_0)}{p(x,z|\theta\_1)} = \prod\_i \frac{p(z\_i|z\_{<i},\theta\_0)}{p(z\_i|z\_{<i},\theta\_1)}$$

.footnote[Credits: Johann Brehmer.]

???

- Computer simulations typically evolve along a tree-like structure of successive random branchings.
- The probabilities of each branching $p(z\\\_i|z\\\_{<i},\theta)$ are often clearly defined in the code:
```python
if random() > 0.1+2.5+model\_parameter:
        do_one_thing()
else:
        do_another_thing()

---

# Rᴀsᴄᴀʟ

.grid[
.kol-1-2[

## Regressing the likelihood ratio

Observe that the joint likelihood ratios
$$r(x,z|\theta\_0, \theta\_1) = \frac{p(x,z|\theta\_0)}{p(x,z|\theta\_1)}$$
are scattered around $r(x|\theta\_0,\theta\_1)$.

Can we use them to approximate $r(x|\theta\_0,\theta\_1)$?
]
.kol-1-2[.width-100[![](figures/lec3/r_xz.png)]]
]

???

The relation between $r(x, z|\theta\_0, \theta\_1)$ and $r(x|\theta\_0, \theta\_1)$ is not trivial— the integral of the ratio is not the ratio of the integrals!

---

class: middle

Consider the squared error of a function $\hat{g}(x)$ that only depends on $x$, but is trying to approximate a function $g(x,z)$ that also depends on the latent $z$:
$$L\_\text{MSE} = \mathbb{E}\_{p(x,z|\theta)} \left[ (g(x,z) - \hat{g}(x))^2 \right].$$

Via calculus of variations, we find that the function $g^\*(x)$ that extremizes $L\_\text{MSE}[g]$ is given by
$$\begin{aligned}
g^\*(x) &= \frac{1}{p(x|\theta)} \int p(x,z|\theta) g(x,z) dz \\\\
&= \mathbb{E}\_{p(z|x,\theta)} \left[ g(x,z) \right]
\end{aligned}$$

---

class: middle

Therefore, by identifying the $g(x,z)$ with the joint likelihood ratio $r(x,z|\theta\_0, \theta\_1)$ and $\theta$ with $\theta\_1$, we define
$$L\_r = \mathbb{E}\_{p(x,z|\theta\_1)} \left[ (r(x,z|\theta\_0, \theta\_1) - \hat{r}(x))^2 \right], $$
which is minimized by
$$
\begin{aligned}
r^\*(x) &= \frac{1}{p(x|\theta\_1)} \int p(x,z|\theta\_1) \frac{p(x,z|\theta\_0)}{p(x,z|\theta\_1)} dz \\\\
&= \frac{p(x|\theta\_0)}{p(x|\theta\_1)} \\\\
&= r(x|\theta\_0,\theta\_1).
\end{aligned}$$

---

<br>

$$r^\*(x|\theta\_0,\theta\_1) = \arg\min\_{\hat{r}} L\_r[\hat{r}]$$

???

How do we solve this minimization problem?

--

count: false

.center.width-80[![](figures/lec3/nn-r.png)]

???


Minimizing functionals is exactly what *machine learning* does!

In our case,
- $\hat{r}$ are neural networks (or the parameters thereof);
- $L\_r$ is the loss function;
- minimization is carried out using stochastic gradient descent from the data extracted from the simulator.

---

class: middle



.grid[
.kol-1-2[
## Regressing the score

Similarly, we can mine the simulator to extract the joint score

$$t(x,z|\theta\\\_0) = \nabla\\\_\theta \log p(x,z|\theta) \big|\_{\theta\_0},$$

which indicates how much more or less likely $x,z$ would be if one changed $\theta\_0$.
]
.kol-1-2[.width-100[![](figures/lec3/t_xz.png)]]
]

???

The integral of the log is not the log of the integral!.

---

class: middle

Using the same trick, by identifying $g(x,z)$ with the joint score $t(x,z|\theta\_0)$ and $\theta$ with $\theta\_0$, we define
$$L\_t = \mathbb{E}\_{p(x,z|\theta\_0)} \left[ (t(x,z|\theta\_0) - \hat{t}(x))^2 \right],$$
which is minimized by
$$\begin{aligned}
t^\*(x) &= \frac{1}{p(x|\theta\_0)} \int p(x,z|\theta\_0) (\nabla\_\theta \log p(x,z|\theta) \big|\_{\theta\_0})  dz \\\\
&= \frac{1}{p(x|\theta\_0)} \int p(x,z|\theta\_0) \frac{\nabla\_\theta p(x,z|\theta) \big|\_{\theta\_0}}{p(x,z|\theta\_0)} dz \\\\
&= \frac{\nabla\_\theta p(x|\theta)\big|\_{\theta\_0}}{p(x|\theta\_0)} \\\\
&= \nabla\_\theta \log p(x|\theta)\big|\_{\theta\_0}\\\\
&= t(x|\theta\_0).
\end{aligned}$$

---

# Rᴀsᴄᴀʟ

$$L\_\text{RASCAL} = L\_r + L\_t$$

.center.width-100[![](figures/lec3/rascal1.png)]

.footnote[Brehmer, Louppe, Pavez and Cranmer, 2018 [[arXiv:1805.12244](https://arxiv.org/pdf/1805.12244.pdf)]]

---

# There is more...

.center.width-100[![](figures/lec3/family.png)]

.footnote[Brehmer, Louppe, Pavez and Cranmer, 2018 [[arXiv:1805.12244](https://arxiv.org/abs/1805.12244)].]

---

# Sᴀʟʟʏ (= optimal compression)

## The local model

In the neighborhood of $\theta\_\text{ref}$, the Taylor expansion of $\log p(x|\theta)$ is
$$\log p(x|\theta) = \log p(x|\theta\_\text{ref}) + \underbrace{\nabla\_\theta \log p(x|\theta)\Big\vert\_{\theta\_\text{ref}}}\_{t(x|\theta\_\text{ref})} \cdot (\theta-\theta\_\text{ref}) + O((\theta-\theta\_\text{ref})^2) $$

.center.width-50[![](figures/lec3/local.png)]

???

Note that the likelihood ratio $r$ relates to the *score*
$$t(x|\theta\_\text{ref}) = \nabla\_\theta \log p(x|\theta)\vert\_{\theta\_\text{ref}} = \nabla\_\theta r(x|\theta,\theta\_\text{ref})\vert\_{\theta\_\text{ref}}.$$
- It quantifies the relative change of the likelihood under infinitesimal changes.
- It can be seen as a **local equivalent of the likelihood ratio**.

---

class: middle

This results in the exponential model
$$p\_\text{local}(x|\theta) = \frac{1}{Z(\theta)} p(t(x|\theta\_\text{ref})|\theta\_\text{ref}) \exp(t(x|\theta\_\text{ref}) \cdot (\theta-\theta\_\text{ref}))$$
where the score $t(x|\theta\_\text{ref})$ are its sufficient statistics.

That is,
- knowing $t(x|\theta\_\text{ref})$ is just as powerful as knowing the full function $\log p(x|\theta)$.
- $x$ can be compressed into a single scalar $t(x|\theta\_\text{ref})$ without loss of power.

.footnote[Brehmer, Louppe, Pavez and Cranmer, 2018 [[arXiv:1805.12244](https://arxiv.org/abs/1805.12244)].]

???

The exponential model is a solution for $p(x|\theta)$ of the Taylor expansion.
with the constraint we seek a probability density function, hence the normalization $Z$.


---

class: middle

## Sᴀʟʟʏ

.center.width-100[![](figures/lec3/sally.png)]

.footnote[Brehmer, Louppe, Pavez and Cranmer, 2018 [[arXiv:1805.12244](https://arxiv.org/abs/1805.12244)].]

---

# Examples

## ① Hunting new physics at particle colliders

The goal is to constrain two EFT parameters and compare against traditional histogram analysis.

.width-60.center[![](figures/lec3/ex1-1.png)]

.footnote[Brehmer, Cranmer, Louppe, and Pavez, 2018a [[arXiv:1805.00020](https://arxiv.org/abs/1805.00020)], 2018b [[arXiv:1805.00013](https://arxiv.org/abs/1805.00013)]; Brehmer, Louppe, Pavez and Cranmer, 2018 [[arXiv:1805.12244](https://arxiv.org/abs/1805.12244)].]

---

class: middle

.center.width-100[![](figures/lec3/ex1-2.png)]

.footnote[Brehmer, Cranmer, Louppe, and Pavez, 2018a [[arXiv:1805.00020](https://arxiv.org/abs/1805.00020)], 2018b [[arXiv:1805.00013](https://arxiv.org/abs/1805.00013)]; Brehmer, Louppe, Pavez and Cranmer, 2018 [[arXiv:1805.12244](https://arxiv.org/abs/1805.12244)].]

???

baseline: 2d histogram analysis of jet momenta and angular correlations

---

class: middle, black-slide

## ② Dark matter substructure from gravitational lensing

.center[.width-45[![](figures/lec3/gl1.png)] .width-50[![](figures/lec3/gl2.png)]]

.footnote[Brehmer, Mishra-Sharma, Hermans, Louppe, and Cranmer, 2019 [[arXiv:1909.02005](https://arxiv.org/abs/1909.02005)].]

???

A dark halo is the inferred halo of invisible material (dark matter) that permeates and surrounds individual galaxies.

A single halo may contain multiple virialized clumps of dark matter bound together by gravity, known as subhalos.

The halo and its constituents have a predictable effect on the lenses.

---

class: middle

.center.width-100[![](figures/lec3/simulations.png)]

Number of dark matter subhalos and their mass and location lead to complex latent space of each image.

The goal is the .bold[inference of population parameters $\beta$ and $f\_\text{sub}$.]

.footnote[Brehmer, Mishra-Sharma, Hermans, Louppe, and Cranmer, 2019 [[arXiv:1909.02005](https://arxiv.org/abs/1909.02005)].]

???

$\beta$ and $f\_\text{sub}$ parameterize the subhalo mass function i.e. the distribution of subhalo masses in a given host halo.


---

class: middle

.center.width-100[![](figures/lec3/dm-inference.png)]

.footnote[Brehmer, Mishra-Sharma, Hermans, Louppe, and Cranmer, 2019 [[arXiv:1909.02005](https://arxiv.org/abs/1909.02005)].]



???

Between the LHC stuff and the lensing, we’re using the same methods across 37 orders of magnitude in length scale.

---

class: middle

## ③ Gravitational waves (work in progress with GRAPPA!)

.center.width-90[![](figures/lec3/freq_contour_1.png)]

---

class: middle
count: false

# Bayesian inference

---

class: middle

.grid[
.kol-1-2[

<br>
Bayesian inference = computing the posterior
$$p(\theta|x) = \frac{p(x|\theta) p(\theta)}{p(x)}.$$

]
.kol-1-2[.width-100[![](figures/lec3/latent-model.svg)]]
]


Doubly **intractable** in the likelihood-free scenario:
- Cannot evaluate the likelihood $p(x|\theta) = \int p(x,z|\theta) dz$.
- Cannot evaluate the evidence $p(x) = \int p(x|\theta)p(\theta) d\theta$.

???

One can learn the likelihood-to-evidence ratio with SL.

---

# Approximate Bayesian Computation (ABC)

.center.width-100[![](figures/lec3/abc.png)]

## Issues

- How to choose $x'$? $\epsilon$? $||\cdot||$?
- No tractable posterior.
- Need to run new simulations for new data or new prior.

.footnote[Credits: Johann Brehmer.]

---

# Amortizing Bayes

The Bayes rule can be rewritten as
$$p(\theta|x) = \frac{p(x|\theta) p(\theta)}{p(x)} = r(x|\theta) p(\theta) \approx \hat{r}(x|\theta)p(\theta),$$
where $r(x|\theta) = \frac{p(x|\theta)}{p(x)}$ is the likelihood-to-evidence ratio.

--

count: false

As before, the likelihood-to-evidence ratio can be approximated e.g. from a neural network classifier trained to distinguish  $x \sim p(x|\theta)$  from $x \sim p(x)$,
hence enabling *direct* and **amortized**  posterior evaluation.
.grid.center[
.kol-1-2[.width-70[![](figures/lec3/aalr-training.png)]]
.kol-1-2[<br>.width-100[![](figures/lec3/aalr-net.png)] ]
]

.footnote[Hermans, Begy and Louppe, 2019 [[arXiv:1903.04057](https://arxiv.org/abs/1903.04057)]; Brehmer, Mishra-Sharma, Hermans, Louppe, and Cranmer, 2019 [[arXiv:1909.02005](https://arxiv.org/abs/1909.02005)].]

???

This is helpful when you known you will have to run posterior inference many times.

---

class: middle

## Bayesian inference of dark matter subhalo population parameters

.width-100[![](figures/lec3/live_inference_with_images_reverse_small.gif)]

.footnote[Brehmer, Mishra-Sharma, Hermans, Louppe, and Cranmer, 2019 [[arXiv:1909.02005](https://arxiv.org/abs/1909.02005)].]

---

# MCMC posterior sampling

<br>
.center.width-100[![](figures/lec3/animation3.gif)]

.footnote[Credits: [Chuck Huber](https://blog.stata.com/2016/11/15/introduction-to-bayesian-statistics-part-2-mcmc-and-the-metropolis-hastings-algorithm/), 2016.]

???

The previous amortization works well enough for small dimensional parameter spaces, but cannot be directly applied to larger ones.

In this scenario, the usual solution is to resort to sampling and marginalization.

Insist that traditional MCMC requires the likelihood to be evaluated, even if it not analytical.

---

class: middle

## Likelihood-free MCMC

MCMC samplers require the evaluation of the posterior ratios:
$$
\begin{aligned}
\frac{p(\theta\_\text{new}|x)}{p(\theta\_{t-1}|x)} &= \frac{p(x|\theta\_\text{new}) p(\theta\_\text{new}) / p(x)}{p(x|\theta\_{t-1}) p(\theta\_{t-1}) / p(x)} \\\\
&= \frac{p(x|\theta\_\text{new}) p(\theta\_\text{new})}{p(x|\theta\_{t-1}) p(\theta\_{t-1})} \\\\
&= r(x|\theta\_\text{new}, \theta\_{t-1}) \frac{p(\theta\_\text{new})}{p(\theta\_{t-1})}
\end{aligned}$$

Again, MCMC samplers can be made *likelihood-free* by plugging a **learned approximation** $\hat{r}(x|\theta\_\text{new}, \theta\_{t-1})$ of the likelihood ratio.

.footnote[Hermans, Begy and Louppe, 2019 [[arXiv:1903.04057](https://arxiv.org/abs/1903.04057)].]

---

class: middle

.width-100[![](figures/lec3/aalr-mcmc.png)]

.footnote[Hermans, Begy and Louppe, 2019 [[arXiv:1903.04057](https://arxiv.org/abs/1903.04057)].]


---

# Summary

- Much of modern science is based on "likelihood-free" simulations.
- The likelihood-ratio is central to many statistical inference procedures, regardless of your religion.
- Supervised learning enables likelihood-ratio estimation.
- Better likelihood-ratio estimates can be achieved by mining simulators.
- (Probabilistic programming enables posterior inference in scientific simulators.)

<br><br>
.center.width-90[![](./figures/lec3/lfi-chain.png)]

---

# Collaborators

<br>

.center.grid[
.kol-1-12[]
.kol-1-6.center[.width-100[![](figures/lec3/faces/kyle.png)] Kyle Cranmer]
.kol-1-6.center[.width-100[![](figures/lec3/faces/juan.png)] Juan Pavez]
.kol-1-6.center[.width-100[![](figures/lec3/faces/johann.png)] Johann Brehmer]
.kol-1-6.center[.width-90[![](figures/lec3/faces/joeri.png)] Joeri Hermans]
.kol-1-6.center[.width-90[![](figures/lec3/faces/antoine.png)] Antoine Wehenkel]
]

.grid[
.kol-1-6.center[.width-100[![](figures/lec3/faces/arnaud.jpg)] Arnaud Delaunoy]
.kol-1-6.center[.width-100[![](figures/lec3/faces/siddarth.png)] Siddarth Mishra-Sharma]
.kol-1-6.center[.width-100[![](figures/lec3/faces/lukas.png)] Lukas Heinrich]
.kol-1-6.center[.width-100[![](figures/lec3/faces/gunes.png)] Atılım Güneş Baydin]
.kol-1-6.center[.width-100[![](figures/lec3/faces/wahid.png)] Wahid Bhimji]
.kol-1-6.center[.width-90[![](figures/lec3/faces/frank.png)] Frank Wood]
]


---

# References


.smaller-xx[
- Brehmer, J., Mishra-Sharma, S., Hermans, J., Louppe, G., Cranmer, K. (2019). Mining for Dark Matter Substructure: Inferring subhalo population properties from strong lenses with machine learning. arXiv preprint arXiv 1909.02005.
- Hermans, J., Begy, V., & Louppe, G. (2019). Likelihood-free MCMC with Approximate Likelihood Ratios. arXiv preprint arXiv:1903.04057.
- Baydin, A. G., Shao, L., Bhimji, W., Heinrich, L., Meadows, L., Liu, J., ... & Ma, M. (2019). Etalumis: Bringing Probabilistic Programming to Scientific Simulators at Scale. arXiv preprint arXiv:1907.03382.
- Stoye, M., Brehmer, J., Louppe, G., Pavez, J., & Cranmer, K. (2018). Likelihood-free inference with an improved cross-entropy estimator. arXiv preprint arXiv:1808.00973.
- Baydin, A. G., Heinrich, L., Bhimji, W., Gram-Hansen, B., Louppe, G., Shao, L., ... & Wood, F. (2018). Efficient Probabilistic Inference in the Quest for Physics Beyond the Standard Model. arXiv preprint arXiv:1807.07706.
- Brehmer, J., Louppe, G., Pavez, J., & Cranmer, K. (2018). Mining gold from implicit models to improve likelihood-free inference. arXiv preprint arXiv:1805.12244.
- Brehmer, J., Cranmer, K., Louppe, G., & Pavez, J. (2018). Constraining Effective Field Theories with Machine Learning. arXiv preprint arXiv:1805.00013.
- Brehmer, J., Cranmer, K., Louppe, G., & Pavez, J. (2018). A Guide to Constraining Effective Field Theories with Machine Learning. arXiv preprint arXiv:1805.00020.
- Casado, M. L., Baydin, A. G., Rubio, D. M., Le, T. A., Wood, F., Heinrich, L., ... & Bhimji, W. (2017). Improvements to Inference Compilation for Probabilistic Programming in Large-Scale Scientific Simulators. arXiv preprint arXiv:1712.07901.
- Louppe, G., Hermans, J., & Cranmer, K. (2017). Adversarial Variational Optimization of Non-Differentiable Simulators. arXiv preprint arXiv:1707.07113.
- Cranmer, K., Pavez, J., & Louppe, G. (2015). Approximating likelihood ratios with calibrated discriminative classifiers. arXiv preprint arXiv:1506.02169.
]

---

class: end-slide, center
count: false

The end.

---

count: false

# Aʟɪᴄᴇ

When the joint likelihood ratio $r(x,z|\theta\_0, \theta\_1)$ is available from the simulator, the corresponding $s(x,z|\theta\_0,\theta\_1)$ are also tractable.

Therefore, the original Cᴀʀʟ cross-entropy can be adapted to make use of the exact $s(x,z|\theta\_0,\theta\_1)$ instead of using labels $y \in \\{ 0,1 \\}$:
$$
\begin{aligned}
L\_{ALICE}[\hat{s}] = -\mathbb{E}\_{p(x,z)} [& s(x,z|\theta\_0,\theta\_1) \log(\hat{s}(x)) + \\\\
&(1-s(x,z|\theta\_0,\theta\_1)) \log(1-\hat{s}(x)) ],
\end{aligned}$$
where $p(x,z) = (p(x,z|\theta\_0) + p(x,z|\theta\_1)) / 2$.

---

count: false

# Probabilistic programming

.center.width-80[![](figures/lec3/sherpa-250.png)]

A probabilistic program defines a joint distribution of *unobserved* $x$ and *observed* $y$ variables $p(x,y)$.

Probabilistic programming extends ordinary programming with two added constructs:
- Sampling from distributions
- Conditioning random variables by specifying observed values

???

What if you also want to do inference on the latent $z$?

---

class: middle
count: false

**Inference engines** give us distributions over unobserved variables, given observed variables (data)
$$p(x|y) = \frac{p(y|x)p(x)}{p(y)}$$

.footnote[Le et al, 2016 [[arXiv:1610.09900](https://arxiv.org/abs/1610.09900)]; Baydin et al, 2018 [[arXiv:1807.07706](https://arxiv.org/abs/1807.07706)]; Baydin et al, 2019 [[arXiv:1907.03382](https://arxiv.org/abs/1907.03382)].]

---

class: middle
count: false

## Probabilistic programming languages

- Anglican (Clojure)
- Church (Scheme)
- **Edward, TensorFlow Probability (Python, TensorFlow)**
- **Pyro (Python, PyTorch)**

---

class: middle, red-slide
count: false

A stochastic simulator implicitly defines a probability distribution by sampling pseudo-random numbers.

.bold[Scientific simulators are probabilistic programs!]

---

class: middle
count: false

.width-80.center[![](figures/lec3/pp-control.png)]

## Key idea

Let a neural network take full control of the internals of the simulation program by hijacking all calls to the random number generator.

.footnote[Le et al, 2016 [[arXiv:1610.09900](https://arxiv.org/abs/1610.09900)]; Baydin et al, 2018 [[arXiv:1807.07706](https://arxiv.org/abs/1807.07706)]; Baydin et al, 2019 [[arXiv:1907.03382](https://arxiv.org/abs/1907.03382)].]

???

Somewhat similar to ABC except we also make inference on $z$ and learn to sample efficiently.

---

class: middle
count: false

.width-100.center[![](figures/lec3/ppx.png)]

.footnote[Le et al, 2016 [[arXiv:1610.09900](https://arxiv.org/abs/1610.09900)]; Baydin et al, 2018 [[arXiv:1807.07706](https://arxiv.org/abs/1807.07706)]; Baydin et al, 2019 [[arXiv:1907.03382](https://arxiv.org/abs/1907.03382)].]

---

class: middle
count: false

.width-80.center[![](figures/lec3/pp-arch.png)]

.footnote[Le et al, 2016 [[arXiv:1610.09900](https://arxiv.org/abs/1610.09900)]; Baydin et al, 2018 [[arXiv:1807.07706](https://arxiv.org/abs/1807.07706)]; Baydin et al, 2019 [[arXiv:1907.03382](https://arxiv.org/abs/1907.03382)].]

---

class: middle
count: false

.grid[
.kol-2-3[
## ③ Taking control of Sherpa (particle physics simulator)
- $\tau$ decay in Sherpa, 38 decay channels, coupled with an approximate calorimeter simulation in C++.
- Observations are 3D calorimeter depositions.
- Latent variables (Monte Carlo truth) of interest: decay channel, px, py, pz momenta, final state momenta and IDs.
]
.kol-1-3[.width-100.center[![](figures/lec3/tau-obs.png)]]
]

.width-100.center[![](figures/lec3/tau.png)]

.footnote[Le et al, 2016 [[arXiv:1610.09900](https://arxiv.org/abs/1610.09900)]; Baydin et al, 2018 [[arXiv:1807.07706](https://arxiv.org/abs/1807.07706)]; Baydin et al, 2019 [[arXiv:1907.03382](https://arxiv.org/abs/1907.03382)].]

---

class: middle
count: false

## Inference results

.grid[
.kol-1-4[.width-100[![](figures/lec3/sherpa-obs.png)]]
.kol-3-4[.width-100[![](figures/lec3/sherpa-inference.png)]]
]

.footnote[Le et al, 2016 [[arXiv:1610.09900](https://arxiv.org/abs/1610.09900)]; Baydin et al, 2018 [[arXiv:1807.07706](https://arxiv.org/abs/1807.07706)]; Baydin et al, 2019 [[arXiv:1907.03382](https://arxiv.org/abs/1907.03382)].]

---

class: middle
count: false

.center[
.width-80[![](figures/lec3/sherpa-posterior-space.png)]

We obtain posteriors over the whole Sherpa address space, 1000s of addresses.]

.footnote[Le et al, 2016 [[arXiv:1610.09900](https://arxiv.org/abs/1610.09900)]; Baydin et al, 2018 [[arXiv:1807.07706](https://arxiv.org/abs/1807.07706)]; Baydin et al, 2019 [[arXiv:1907.03382](https://arxiv.org/abs/1907.03382)].]

---

class: middle
count: false

## Interpretability

Latent probabilistic structure of the 250 most frequent trace types:

.width-100[![](figures/lec3/sherpa-250.png)]

.footnote[Le et al, 2016 [[arXiv:1610.09900](https://arxiv.org/abs/1610.09900)]; Baydin et al, 2018 [[arXiv:1807.07706](https://arxiv.org/abs/1807.07706)]; Baydin et al, 2019 [[arXiv:1907.03382](https://arxiv.org/abs/1907.03382)].]

---

class: middle
count: false

.width-100[![](figures/lec3/sherpa-posterior.png)]

.footnote[Le et al, 2016 [[arXiv:1610.09900](https://arxiv.org/abs/1610.09900)]; Baydin et al, 2018 [[arXiv:1807.07706](https://arxiv.org/abs/1807.07706)]; Baydin et al, 2019 [[arXiv:1907.03382](https://arxiv.org/abs/1907.03382)].]
