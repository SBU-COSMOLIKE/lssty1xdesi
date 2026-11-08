# The pre-implementation recommendation of notes/higher_order_bias, kept verbatim (LaTeX source) for reference

## Section "Recommendation"

```latex
\section{Recommendation}
\label{sec:recommendation}

Both models below are one-loop models.  Each carries the complete
real-space one-loop Eulerian bias set of Eq.~\eqref{eq:pgg} ($\bone$,
$\btwo$, $\bs$, $\bthree$), the $k^2$ counterterm $\alpha_0$ and the
stochastic amplitude $\mathrm{SN}_0$; they differ in what is free.
Model A ties the higher-order biases to the calibrations through the
per-bin linear bias, with group offsets under modest priors; model B
frees them and adds the redshift-space terms.

\textbf{At $\kmax = 0.075\hMpc$} (the cut matched to the thin-bin
radial resolution), the minimal defensible model per DESI chunk,
called model A below, is
\begin{itemize}
\item $b_{1,i} = B_g(1+\delta_i)/D(z_i)$ per thin bin, the memory
model of the correlated-prior note
(Eqs.~\eqref{eq:linear-perbin} and \eqref{eq:kernel}): the amplitude
$B_g$ and the tilt $s_g$ free with flat priors, one each per group,
twelve parameters in total, and the residuals $\varepsilon_i$ under
the exponential-kernel prior.  The baseline sets $\sigma_g = 3\%$ for
BGS, sized to its flux-limit curvature, $\sigma_g = 1\%$ for the
other groups, and $L = 0.2$.  The pessimistic choice sets
$\sigma_g = 3\%$ in every group and $L = 0.05$, the redshift scale of
local selection features such as the 4000\,\AA\ break crossing a
filter edge;
\item $b_{2,i}$ from the Lazeyras relation on $b_{1,i}$
(Eq.~\eqref{eq:perbin-evolution}) plus a free
offset $\Delta_{2,g}$, Gaussian prior of width $1.0$ centered on
zero;
\item $b_{s^2,i}$ from co-evolution on $b_{1,i}$ plus a free offset
$\Delta_{s,g}$, Gaussian prior of width $0.5$ centered on zero (the
width covers the co-evolution-versus-excursion-set spread at LRG
biases);
\item $\bthree$ fixed to co-evolution (or zero; the difference is
inside the $\alpha_0$ freedom);
\item one counterterm $\alpha_0$ per chunk, Gaussian prior centered on
zero with the DESI rule rescaled to this cut: a $50\%$ correction at
$\kmax = 0.075\hMpc$ means a width of
$0.5/\kmax^2 \simeq 89\,h^{-2}\mathrm{Mpc}^2$;
\item one stochastic amplitude $\mathrm{SN}_0$ per chunk, Gaussian
prior of width twice the Poisson power, following DESI;
\item the sampled $\sigma_v$ groups kept as the radial damping
freedom, with the implementation study measuring whether they cover
the dropped $\mu$-dependent terms at this cut.
\end{itemize}
The linear sector carries one amplitude per thin bin (76 in the
shipped layout of equal comoving bin width
$\Delta d_{\rm com} = 62$\,Mpc), constrained by the kernel prior in
every direction except the flat amplitude $B_g$ and tilt $s_g$ of
each group: twelve flat directions, with the remaining 64 residual
combinations held by the prior.  The 24 higher-order parameters (four
per chunk: $\Delta_{2,g}$, $\Delta_{s,g}$, $\alpha_0$,
$\mathrm{SN}_0$) come on top, against DESI DR1's seven per tracer at
$\kmax = 0.2\hMpc$; the pessimistic option of Section~\ref{sec:ladder}
tilts the two offsets and $\alpha_0$ and reaches 42, seven per chunk
(Table~\ref{tab:ladder}).  The reasoning: at
this cut every correction is $1$--$3\%$ with mild scale dependence, so
calibration-centered Gaussian freedom is enough, and the offsets keep
the model honest about the calibrations without opening flat
directions.

The growth-tied linear bias $\bone(z) = B_g/D(z)$ with no deviations,
the shape the shipped parameter files tie to, is not a defensible
alternative.  The correlated-prior note shows that even a conserved
population departs from that shape, by about $1\%$ per
$\Delta z = 0.05$ through the bias continuity equation, and that
selected samples drift more: the BGS flux limit moves the bias by
about $22\%$ across its group (its sections ``The bias continuity
equation, step by step'' and ``The BGS flux-limit bias model, step by
step'').  The linear-bias fits of \texttt{notes/linear\_bias\_kmax}
measure what the
tying buys in the dark-energy figure of merit ${\rm FoM}(w_0, w_a)$,
the inverse area of the error ellipse of the equation-of-state
parameters $w_0$ and $w_a$, on DESI $C_\ell^{gg}$ alone at
$\kmax = 0.075\hMpc$ and the $\Delta d_{\rm com} = 62$\,Mpc layout:
the growth-tied prior gives $32.9$, the baseline prior $18.2$,
the pessimistic prior $11.4$, and the free prior, one free amplitude
per bin, $10.7$.  The factor of $1.8$ between the growth-tied and
baseline-prior numbers rests on a rigidity the physics does not
support: in the accuracy test of the linear-bias priors (defined in
the correlated-prior note's section ``The accuracy test: the figure of
bias''), a conserved galaxy population
fitted with the growth-tied prior
displaces $(w_0, w_a)$ by $11$--$21\sigma$, while the baseline and
pessimistic priors keep the displacement below $\sigma/3$.

\textbf{At $\kmax \simeq 0.15$--$0.2\hMpc$} (the shipped mask), the
model, called model B below, keeps the per-bin linear bias
$b_{1,i}$ with the memory model and widens the higher-order
sector: the $\btwo$ and $\bs$ offsets with the calibration relations
demoted to prior centers (widths of order 2 and 1), $\alpha_0$ at
the DESI DR1 width $12.5\,h^{-2}\mathrm{Mpc}^2$, an explicit radial
counterterm (the $\alpha_2\mu^2 k^2\plin$ term projected through the
$j_\ell''$ transfer, or a validated $\sigma_v$-based equivalent) at
the same DESI DR1 width, $\mathrm{SN}_0$ at twice the Poisson power,
$\mathrm{SN}_2$ at the DESI DR1 width of five units of
$f_{\rm sat}\sigma_v^2/\bar{n}$ (Section~\ref{sec:desi}), and the
redshift-space one-loop question settled by measurement rather than
assumption.  Counting the linear amplitude and tilt, the baseline
option is eight parameters per chunk: the DESI DR1 budget of seven
plus one $\bone$ tilt; the pessimistic option also tilts the offsets
and both counterterms (Table~\ref{tab:ladder}).  The honest
statement is that the real-space-loop shortcut is unvalidated at this
cut: until a Fisher-bias test shows otherwise, headline numbers from
the shipped mask carry a modeling systematic of the size of the
$\alpha_2$-equivalent terms.  Sampling in a physical basis
($\bone\sigma_8$-like combinations) is worth adopting at either cut,
for DESI's reason: it tracks what the data constrain and reduces
projection effects.

In both configurations, the fiducial higher-order values are the
co-evolution and Lazeyras values of Figure~\ref{fig:coev}, not imports
of DESI chain values, for the reason
Section~\ref{sec:verdict} ends with; and the covariance and Fisher
machinery must adopt the same model as the data vector, including the
stochastic sector.
```

## Subsection "Model B and the order of work"

```latex
\subsection{Model B and the order of work}

Model B widens the $\btwo$ and $\bs$ offset priors to the demoted
calibrations (yaml work once model A is in) and adds the radial
counterterm $\alpha_2$, projected with the $j_\ell''$ transfer like
the Kaiser term, at the DESI DR1 width $12.5\,h^{-2}\mathrm{Mpc}^2$,
the stochastic term $\mathrm{SN}_2$ at the DESI DR1 width of five
units of $f_{\rm sat}\sigma_v^2/\bar{n}$, and the redshift-space
one-loop terms.  CosmoLike has no redshift-space
loop templates and FAST-PT's redshift-space terms are not in the C
port, so the options are: absorb them in the sampled $\sigma_v$ plus
an explicit $\alpha_2$-like counterterm and measure the Fisher bias
against a full redshift-space model, or port the FAST-PT
redshift-space templates (or an external $P(k,\mu)$ emulator).
Model B is a research task, not an extension of model A.

The recommended order:
\begin{enumerate}
\item yamls: a grouping that samples every \texttt{DESI\_B1\_}$n$
and one tilt per group, plus the cobaya external prior carrying the
kernel Gaussian on the residuals for chains;
\item interface: free the tidal argument (co-evolution default), add
the $\mathrm{SN}_0$ slot, state the $b_K$-to-$\alpha_0$ mapping;
\item likelihood: the $\btwo$ and $\bs$ trends on $b_{1,i}$ with
group offsets, $\alpha_0$, $\mathrm{SN}_0$, with the priors of
Section~\ref{sec:recommendation};
\item C: one-loop for the adjacent cross pairs through the separable
$D^2(z_1)D^2(z_2)$ templates;
\item validation: CosmoLike's Limber one-loop against CCL's
\texttt{EulerianPTCalculator} with the same templates and the same
\texttt{sub\_lowk} choice (the \texttt{ccl} conda environment carries
pyccl and FAST-PT; \texttt{cocoapy311} carries neither), then the
low-rank non-Limber against a brute-force Bessel integral for one
thin pair;
\item covariance: the one-loop $P_{gg}$ and the $B_g/D(z)$ bias fix
together;
\item model B only after a measured Fisher bias of the
redshift-space loops.
\end{enumerate}
```
