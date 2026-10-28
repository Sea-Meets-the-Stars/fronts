# Note to Lauren: sign attrs on the chunk-store surface fluxes (draft; JXP to forward)

**Store:** `s3://dbof/LLC4320_RAW/CHUNKS/monterey_bay/{YYYYMMDDTHH}.zarr` (Nautilus), the 72-hour
transfer `run_chunks_monterey_72h.yaml` (2012-07-02 T00 → 07-04 T23). Thank you for the transfer.
Everything we need is there, and it is bit-identical to OSN at `k = 0`.

**What the attrs say.** The three flux variables carry these attrs:
- `oceQnet`: "net surface heat flux into the ocean **(+=down)**, >0 increases theta"
- `oceQsw`: "net Short-Wave radiation **(+=down)**, >0 increases theta"
- `oceFWflx`: "net surface Fresh-Water flux into the ocean **(+=down)**, >0 decreases salinity"

**What the data show.** The values are **upward-positive**, the opposite of the attrs.
The numbers are tile means over tile 330 (face 10, j 0:720, i 2880:3600), for the 24 hours of
2012-07-03:
- `oceQsw` is **≤ 0 at every ocean pixel in every hour**; there is no positive value anywhere.
  Its tile mean runs from −0.1 W m⁻² at 09 UTC (≈01 local solar time) to **−589 W m⁻² at
  21 UTC (≈13 local)**. At 2012-07-02 00 UTC it spans −454 to −252 W m⁻². Downward-positive
  shortwave would be ≥ 0 and peak at local noon.
- `oceQnet` is about **+115 W m⁻² at night** and **−454 W m⁻² at 21 UTC**, i.e. positive when
  the ocean loses heat.
- `oceFWflx` is about **+2 to +3 × 10⁻⁵ kg m⁻² s⁻¹** in the tile mean, i.e. positive for net
  evaporation over a summer subtropical ocean.

All three are consistent with MITgcm's *forcing* convention (`Qnet`, `Qsw` and `EmPmR` are
positive upward, out of the ocean). The attrs are the text of the *diagnostics* package's
`oceQnet`/`oceQsw`/`oceFWflx`, which define the opposite sign. So the long_names do not match
the numbers in the store. We have not traced where the attrs are attached: the transfer code,
dbof's variable metadata, or xmitgcm's.

**Why it matters.** Anyone who takes the long_name at face value flips the sign of the
surface heat and freshwater forcing. For us, that would have flipped the diabatic term of the
frontogenesis budget.

**Suggested fix (your call).** Either:
- (a) correct the attrs to the upward convention, e.g. "net upward surface heat flux
  (+=up), >0 decreases theta", and the analogues for `oceQsw` and `oceFWflx`; or
- (b) negate the data in the transfer so that the existing attrs become true.

(a) is cheaper and does not change any bytes already written. Either way, it is worth
checking the other chunk regions (`gulf_stream`, `ross`, …), which presumably share the code
path, and the 7 daily stores outside our window.

**What we did on our side.** Our loader (`vertical.load_chunk_levels`, frontogenesis M2) negates
all three fields at write time, so our local store is **downward-positive**, as our budget code
assumes. Each variable carries a correct `sign_convention` attr, the original long_name is kept
as `source_sign_convention`, and the negation is recorded in the store's provenance. The loader
also refuses an hour whose raw `oceQsw` exceeds +1 W m⁻². So if the source is ever corrected by
negating the data (b), our pull fails loudly instead of flipping the sign twice. Please let us
know if you change it.

Side note, needing no action: the fluxes are piecewise linear with kinks every 6 h, at
03/09/15/21 UTC. This is consistent with 6-hourly atmospheric forcing interpolated by the model,
so the hourly shortwave is a triangle rather than resolved insolation.
