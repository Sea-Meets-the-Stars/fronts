# Vet-idea report on the M3 interim report (LLC4320 surface frontogenesis)

Date of review: 26 October 2026 Document vetted: `frontogenesis_M3_interim.md` (Prochaska, with Claude; dated 12 October 2026\)

## 1\. Framing

The M3 interim report claims that hourly, roughly 2 km surface output from LLC4320 cannot close the surface budget for front strength G \= \|∇b\|². The measured tendency D\_hG/Dt (a one-hour semi-Lagrangian step) is compared with the predicted 2F, minus a sub-filter term (τ), a vertical term (V), and a surface-flux term (S). On one tile off Monterey Bay over 72 summer hours, the residual is 0.81 to 1.04 of the measured amplitude against a pre-declared tolerance of 0.50. The report attributes this to the hourly cadence at the grid scale and to a surface-flux term that is nearly uncorrelated with the imbalance at filtered scales.

Claims the idea depends on:

1. C1. Novelty and value of this null as a publishable result.  
2. C2. Mechanism and budget bookkeeping, including the tilting term reintroduced by the 1 m top cell of the linear free surface.  
3. C3. Hourly output limits the tendency estimate.  
4. C4. Numerical diffusion of the 7th-order flux-limited scheme is scale-selective and lives in the residual.  
5. C5. KPP mixes surface flux through the boundary layer within the hour, so S should scale with h\_bl, and S legitimately belongs in the budget.  
6. C6. Methods (semi-Lagrangian D/Dt, coarse-graining, midpoint front selection, block bootstrap).  
7. C7. The failure is a property of the data rather than of the analysis.

## 2\. Verdict

**Proceed with changes.** The report's logic is sound and its pre-declared tolerance is a real strength. The claim that hourly 2 km surface fields cannot constrain the budget is not yet earned, because the analysis has no demonstrated detection limit and the causal attributions are inferences. The literature search found no direct precedent, which fits novelty but also fits "tried and unpublished."

## 3\. Strongest objection

The null has no independent noise floor. The semi-Lagrangian and Eulerian estimates of the same tendency differ by 0.672 and 0.408 of the amplitude at L \= 0 and 2, which is comparable to the 0.50 tolerance. The discrete null (0.981) only checks the kinematic operator, not how the hourly step interacts with true sub-hourly evolution. Until the pipeline is run on a case with a known answer, "the data cannot do it" and "this analysis cannot do it" are indistinguishable.

How to answer it: advect a passive tracer offline with the model's hourly velocities and run the full budget on it, or use a synthetic front with known forcing. Also report the ratio for a time-shuffled control so that "no skill" has a number.

## 4\. Claims table

| Claim | Status | Key references and notes |
| :---- | :---- | :---- |
| C1 Novelty of this null | Unverified. No direct precedent found. | See section 5\. |
| C2 Mechanism and bookkeeping | Mostly supported. Top-cell reintroduction of tilting has no literature support. | Hoskins & Bretherton 1972; McWilliams 2021; Verma, Pham & Sarkar. |
| C3 Hourly output limits the tendency estimate | Contested as worded. Tidal and near-inertial periods sit above the roughly 2 h Nyquist period, so only sub-2 h waves alias. | Torres et al. 2018; Maalouly et al. |
| C4 Scale-selective numerical diffusion lives in R | Partly supported. Nothing quantifies it, and the red team calls the attribution circular because partial correlations stay below 0.22. | NASA S-MODE README; MITgcm manual; McWilliams 2021\. |
| C5a KPP mixes flux within the hour, so S scales with h\_bl | Unverified in the literature. The data support it for magnitude only. | Large et al. 1994\. |
| C5b Surface flux belongs in a frontogenesis budget | Supported qualitatively. | Ohishi 2019; Dauhajre 2025; McWilliams 2017; Cai 2025\. |
| C6 Methods | Coarse-graining, block bootstrap and regression dilution supported. No source found for semi-Lagrangian D/Dt. | Germano; Eyink & Aluie; Aluie et al.; Künsch; Frost & Thompson. |
| C7 Failure is a property of the data | Not accepted until the noise-floor test is done. | None. |

## 5\. Closest prior work

1. Srinivasan, Barkan & McWilliams 2023 links forward energy flux to frontogenesis via coarse-graining. It has no buoyancy-gradient closure.  
2. Yu, Barkan & Naveira Garabato 2024 measures frontogenesis at about 2 km from moorings. It is observational.  
3. Dauhajre et al. 2025 covers vertical mixing inducing or inhibiting sharpening in idealized 2D fronts.  
4. Verma, Pham & Sarkar is an LES squared-gradient budget that is unforced and has no hourly-output limit.  
5. Capet et al. 2008 Part II treats frontal processes in a ROMS California Current run. What it budgets was not confirmed.  
6. Peng & Bodner is an analytical model using D/Dt of (∂b/∂x)² as its diagnostic.

## 6\. Suggested changes, ranked

1. **Add the noise-floor test** (offline passive tracer or synthetic front) plus a time-shuffled control. This answers the strongest objection directly.  
2. **Downgrade the sampling claim to a hypothesis.** At L \= 8 the estimator gap is only 0.183 while R is 1.04, so sampling cannot be the whole story at filtered scales. The report already says this. The red team's remark that L \= 16 "should have recovered closure" misreads the two-cause structure, but the wording point stands.  
3. **Test V against a different top cell.** Recompute it from levels two and three. If it moves, it is a grid artefact. V is only 1.5 to 1.7% of the amplitude, so the concern that adding V and S worsens R really falls on S.  
4. **Split S** into turbulent flux (mixed over h\_bl) and shortwave (with the attenuation profile), and check whether R falls. The red team asked whether S is built from the flux gradient. Section 2.5 of the report says it is the ∇b·∇ contraction of the tendency, so that part appears answered.  
5. **Address selection and pooling.** Select front pixels from a different variable or time, resample in time with 12 to 24 h blocks, and run a second tile or season before generalizing.

Further observation from the red team: the residual is positive at every local hour, which looks like a systematic bias and not diffusion. The residual slope on 2F of \+0.44 to \+0.53 (measured exceeds 2F) points the same way. The report currently logs the diurnal result as an unmet expectation.

## 7\. Where the two reviewers' readings should be weighed

- The red team's point about the followed front is fair. The four-term integral reaching 0.99 of the observed change while 2F alone gives 8.47e-14 against 1.55e-14 observed is a path-integrated fit and not evidence that the terms are right. Regression to the mean after selecting the 90th percentile is a competing explanation for part of the drop, although the unselected control (0.89) argues against it being the whole story.  
- The red team's claim that 0.81 to 1.04 is "what random noise of the same variance would give" is speculative. The shuffled-in-time control in change 1 would settle it.

## 8\. Citations

Status key: **\[full\]** read in full, **\[abstract only\]**, **\[metadata only\]** (citation confirmed, content not read). A warning mark means the red team's spot-check found a problem.

- Aluie, H., Hecht, M., Vallis, G. K. (2018). Mapping the energy cascade in the North Atlantic Ocean: the coarse-graining approach. J. Phys. Oceanogr. [https\://doi.org/10.1175/jpo-d-17-0100.1](https://doi.org/10.1175/jpo-d-17-0100.1) \[abstract only\]  
- Barkan, R., Molemaker, M. J., Srinivasan, K., McWilliams, J. C., D'Asaro, E. A. (2019). The role of horizontal divergence in submesoscale frontogenesis. J. Phys. Oceanogr. [https\://doi.org/10.1175/jpo-d-18-0162.1](https://doi.org/10.1175/jpo-d-18-0162.1) \[metadata only\]  
- Cai, Y., Lei, R., Chen, D., et al. (2025). Wind-front interaction within the oceanic mixed layer following a storm in the Southern Ocean. Adv. Atmos. Sci. 42, 2399–2410. [https\://doi.org/10.1007/s00376-025-4361-5](https://doi.org/10.1007/s00376-025-4361-5) \[abstract only\]  
- Capet, X., McWilliams, J. C., Molemaker, M. J., Shchepetkin, A. F. (2008). Mesoscale to submesoscale transition in the California Current System. Part II: Frontal processes. J. Phys. Oceanogr. [https\://doi.org/10.1175/2007jpo3672.1](https://doi.org/10.1175/2007jpo3672.1) \[metadata only\]  
- Chereskin, T. K., et al. (2019). Characterizing the transition from balanced to unbalanced motions in the Southern California Current. J. Geophys. Res. Oceans. [https\://doi.org/10.1029/2018jc014583](https://doi.org/10.1029/2018jc014583) \[metadata only; the LLC4320 summer IGW claim attached to it is secondhand\]  
- Dauhajre, D. P., Srinivasan, K., Molemaker, M. J., Gula, J., Hypolite, D., McWilliams, J. C., Barkan, R., Young, W. R. (2025). Vertical mixing can both induce and inhibit submesoscale frontogenesis. J. Phys. Oceanogr. [https\://doi.org/10.1175/jpo-d-24-0148.1](https://doi.org/10.1175/jpo-d-24-0148.1) (preprint [https\://doi.org/10.22541/essoar.172312989.92079130/v1](https://doi.org/10.22541/essoar.172312989.92079130/v1)) \[abstract only\]  
- Eyink, G. L., Aluie, H. (2009). Localness of energy cascade in hydrodynamic turbulence. I. Smooth coarse graining. Phys. Fluids. [https\://doi.org/10.1063/1.3266883](https://doi.org/10.1063/1.3266883)  
- Frost, C., Thompson, S. G. (2000). Correcting for regression dilution bias: comparison of methods for a single predictor variable. J. R. Stat. Soc. A 163, 173–189. [https\://doi.org/10.1111/1467-985X.00164](https://doi.org/10.1111/1467-985X.00164) \[abstract only\]  
- (Warning) Gallmeier, K., Prochaska, J. X., Cornillon, P., Menemenlis, D., Kelm, M. (2023). An evaluation of the LLC4320 global-ocean simulation based on the submesoscale structure of modeled sea surface temperature fields. Geosci. Model Dev. 16, 7143\. [https\://doi.org/10.5194/gmd-16-7143-2023](https://doi.org/10.5194/gmd-16-7143-2023) (preprint arXiv:2303.13949) \[abstract only by the researcher. The red team could not open it, so the "below O(10) km" paraphrase is unconfirmed.\]  
- Germano, M. (1992). Turbulence: the filtering approach. J. Fluid Mech. 238, 325–336. [https\://doi.org/10.1017/S0022112092001733](https://doi.org/10.1017/S0022112092001733) \[abstract only\]  
- Hoskins, B. J., Bretherton, F. P. (1972). Atmospheric frontogenesis models: mathematical formulation and solution. J. Atmos. Sci. 29, 11–37. [https\://doi.org/10.1175/1520-0469(1972)029](https://doi.org/10.1175/1520-0469\(1972\)029)\<0011:afmmfa\>2.0.co;2 \[metadata only\]  
- Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. Ann. Stat. 17(3). [https\://doi.org/10.1214/aos/1176347265](https://doi.org/10.1214/aos/1176347265) \[metadata only\]  
- Large, W. G., McWilliams, J. C., Doney, S. C. (1994). Oceanic vertical mixing: a review and a model with a nonlocal boundary layer parameterization. Rev. Geophys. 32, 363–403. [https\://doi.org/10.1029/94RG01872](https://doi.org/10.1029/94RG01872) \[abstract only\]  
- Maalouly, M., Dekens, A., Lapeyre, G., Ponte, A. L. S., Berti, S. Effects of high-frequency and balanced motions on Lagrangian pair dispersion at the ocean surface. Preprint. [https\://arxiv.org/abs/2506.01002](https://arxiv.org/abs/2506.01002) \[abstract only\]  
- McWilliams, J. C. (2017). Submesoscale surface fronts and filaments: secondary circulation, buoyancy flux, and frontogenesis. J. Fluid Mech. 823, 391–432. [https\://doi.org/10.1017/jfm.2017.294](https://doi.org/10.1017/jfm.2017.294) \[abstract only\]  
- McWilliams, J. C. (2021). Oceanic frontogenesis. Annu. Rev. Mar. Sci. 13, 227–253. [https\://doi.org/10.1146/annurev-marine-032320-120725](https://doi.org/10.1146/annurev-marine-032320-120725) \[abstract only\]  
- Ohishi, S., Aiki, H., Tozuka, T., Cronin, M. F. (2019). Frontolysis by surface heat flux in the eastern Japan Sea: importance of mixed layer depth. J. Oceanogr. 75, 283–297. [https\://doi.org/10.1007/s10872-018-0502-0](https://doi.org/10.1007/s10872-018-0502-0) \[abstract only\]  
- Peng, S., Bodner, A. (2026). A theoretical model for oceanic submesoscales under next-order effects of strain and turbulence. Preprint. [https\://arxiv.org/abs/2601.18962](https://arxiv.org/abs/2601.18962) \[abstract only\]  
- Rocha, C. B., Gille, S. T., Chereskin, T. K., Menemenlis, D. (2016). Seasonality of submesoscale dynamics in the Kuroshio Extension. Geophys. Res. Lett. [https\://doi.org/10.1002/2016gl071349](https://doi.org/10.1002/2016gl071349) \[metadata only\]  
- Soufflet, Y., et al. (2016). On effective resolution in ocean models. Ocean Modelling. [https\://doi.org/10.1016/j.ocemod.2015.12.004](https://doi.org/10.1016/j.ocemod.2015.12.004) \[metadata only\]  
- Srinivasan, K., Barkan, R., McWilliams, J. C. (2023). A forward energy flux at submesoscales driven by frontogenesis. J. Phys. Oceanogr. 53(1), 287–305. [https\://cris.iucc.ac.il/en/publications/a-forward-energy-flux-at-submesoscales-driven-by-frontogenesis-2/](https://cris.iucc.ac.il/en/publications/a-forward-energy-flux-at-submesoscales-driven-by-frontogenesis-2/) (preprint DOI 10.1002/essoar.10509991.2; journal DOI not confirmed) \[abstract only\]  
- Su, Z., et al. (2018). Ocean submesoscales as a key component of the global heat budget. Nat. Commun. [https\://doi.org/10.1038/s41467-018-02983-w](https://doi.org/10.1038/s41467-018-02983-w) \[metadata only\]  
- (Warning) Torres, H. S., Klein, P., Menemenlis, D., Qiu, B., Su, Z., Wang, J., Chen, S., Fu, L.-L. (2018). Partitioning ocean motions into balanced motions and internal gravity waves. J. Geophys. Res. Oceans 123, 8084–8105. [https\://doi.org/10.1029/2018JC014438](https://doi.org/10.1029/2018JC014438) (record confirmed at [https\://archimer.ifremer.fr/doc/00464/57568](https://archimer.ifremer.fr/doc/00464/57568)) \[The paper is confirmed. The researcher's statements about hourly sampling with a 2 h Nyquist period and little SST and SSS signal were not confirmed by the page the red team read. Check the full text before citing either.\]  
- Verma, V., Pham, H. T., Sarkar, S. The submesoscale, the finescale and their interaction at a mixed layer front. Preprint. [https\://arxiv.org/abs/1910.07625](https://arxiv.org/abs/1910.07625) \[abstract only\]  
- Yu, X., Barkan, R., Naveira Garabato, A. C. (2024). Intensification of submesoscale frontogenesis and forward energy cascade driven by upper-ocean convergent flows. Nat. Commun. 15\. [https\://doi.org/10.1038/s41467-024-53551-4](https://doi.org/10.1038/s41467-024-53551-4) \[abstract only\]  
- MITgcm manual, advection schemes. [https\://mitgcm.org/public/r2\_manual/final/online\_documents/node82.html](https://mitgcm.org/public/r2_manual/final/online_documents/node82.html)  
- NASA S-MODE model description (child grid with LLC4320 settings; lists Leith viscosity and the 7th-order scheme). [https\://data.nas.nasa.gov/smode/smodedata/README/model\_description.pdf](https://data.nas.nasa.gov/smode/smodedata/README/model_description.pdf)

### Not verified or not read

- Hoskins (1982) was not looked up.  
- arXiv 2608.14955 (generative data assimilation and fronts) returned no text.  
- Gula's filament work was found only as a project page.  
- The Srinivasan et al. 2023 journal DOI is unconfirmed. Only the ESSOAR preprint DOI was verified.

## 9\. Open gaps from the literature search

- Nothing in the literature on a linear free surface with a 1 m top cell reintroducing the vertical (tilting) term. This appears to be the authors' own reasoning.  
- No methods reference found for a one-hour semi-Lagrangian estimate of D/Dt.  
- No source quantifies the scale-selectivity of the 7th-order limited scheme or shows it dominates front dissipation in LLC4320.  
- Explicit tracer diffusion in the primary LLC4320 documentation was not checked. The only description read was a child-grid README that lists a modified Leith scheme for horizontal viscosity.

## 10\. Decision

Approve as it stands, revise (give direction), or dig deeper. Obvious deeper targets are the full text of Torres 2018 and Gallmeier 2023, a search for a semi-Lagrangian D/Dt methods reference, and the Srinivasan 2023 journal DOI. Gallmeier et al. 2023 has Prochaska as a coauthor, so he may be able to settle the resolution question quickly.