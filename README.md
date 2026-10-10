# Multi-turn history representation ablation

History Selection has been rerun using the recall-oriented selector prompt (recall_v1). Raw Multi-turn and One-line Rewrite results are unchanged.

Every sample receives the same source document. History uses gold previous answers.
Scores include failures as incorrect. Token/latency metrics only include successful calls.

| Suite | Variant | n | Accuracy/EM | Token F1 | Mean answer input tokens | Mean TOTAL tokens | Mean answer sec | Mean TOTAL sec | Mean kept turns | Errors | Completed accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| easy_coqa | raw_multiturn | 100/100 | 0.320 | 0.559 | 522.8 | 557.5 | 1.54 | 1.54 | 4.43 | 0 | 0.320 |
| easy_coqa | history_selection | 100/100 | 0.290 | 0.512 | 490.0 | 1047.7 | 1.62 | 4.09 | 2.80 | 0 | 0.290 |
| easy_coqa | one_line_rewrite | 100/100 | 0.290 | 0.500 | 445.0 | 765.0 | 1.34 | 3.67 | 0.00 | 0 | 0.290 |
| hard_convfinqa | raw_multiturn | 100/100 | 0.770 | 0.362 | 1066.4 | 1182.4 | 2.06 | 2.06 | 3.26 | 0 | 0.770 |
| hard_convfinqa | history_selection | 100/100 | 0.780 | 0.341 | 1054.8 | 1662.0 | 2.21 | 5.20 | 2.79 | 0 | 0.780 |
| hard_convfinqa | one_line_rewrite | 100/100 | 0.760 | 0.366 | 1002.8 | 1388.0 | 1.83 | 3.75 | 0.00 | 0 | 0.760 |

*Easy accuracy = normalized Exact Match; hard accuracy = executed-answer numeric match (relative tolerance 1%, or absolute 1e-4).
*Total seconds are per-sample sequential API wall times (prep + answer), not wall time with concurrent workers. Total tokens include both LLM stages.
*The primary Accuracy column treats pipeline errors as wrong (end-to-end operational score). Completed accuracy excludes API/preprocessing failures and is not directly comparable when failure rates differ.
*Mean kept turns counts Q/A pairs seen by the answer model: all for Raw, chosen for Selection, and zero for Rewrite.

## Paired wins / losses (same samples)

| Suite | Comparison | Win | Tie | Loss | Δ accuracy | 95% paired bootstrap CI |
|---|---|---:|---:|---:|---:|---:|
| easy_coqa | history_selection vs raw_multiturn | 2 | 93 | 5 | -0.030 | [-0.090, +0.020] |
| easy_coqa | one_line_rewrite vs raw_multiturn | 2 | 93 | 5 | -0.030 | [-0.080, +0.020] |
| easy_coqa | history_selection vs one_line_rewrite | 2 | 96 | 2 | +0.000 | [-0.040, +0.040] |
| hard_convfinqa | history_selection vs raw_multiturn | 5 | 91 | 4 | +0.010 | [-0.050, +0.070] |
| hard_convfinqa | one_line_rewrite vs raw_multiturn | 7 | 85 | 8 | -0.010 | [-0.090, +0.060] |
| hard_convfinqa | history_selection vs one_line_rewrite | 8 | 86 | 6 | +0.020 | [-0.050, +0.090] |

## Paired wins / losses — both completed (answer quality only)

| Suite | Comparison | n paired | Win | Tie | Loss | Δ accuracy |
|---|---|---:|---:|---:|---:|---:|
| easy_coqa | history_selection vs raw_multiturn | 100 | 2 | 93 | 5 | -0.030 |
| easy_coqa | one_line_rewrite vs raw_multiturn | 100 | 2 | 93 | 5 | -0.030 |
| easy_coqa | history_selection vs one_line_rewrite | 100 | 2 | 96 | 2 | +0.000 |
| hard_convfinqa | history_selection vs raw_multiturn | 100 | 5 | 91 | 4 | +0.010 |
| hard_convfinqa | one_line_rewrite vs raw_multiturn | 100 | 7 | 85 | 8 | -0.010 |
| hard_convfinqa | history_selection vs one_line_rewrite | 100 | 8 | 86 | 6 | +0.020 |

Bootstrap intervals resample matching question pairs (2,000 resamples). This report contains up to 100 examples per dataset; check paired n and error counts.
For small smoke tests, confidence intervals are descriptive only.

## Selected disagreements

- **easy_coqa** / `coqa:190:6` — Question: Where was that? / Gold: `Los Angeles` / raw=0, selection=1, rewrite=0
- **easy_coqa** / `coqa:265:5` — Question: Did they look alike? / Gold: `Yes` / raw=1, selection=0, rewrite=0
- **easy_coqa** / `coqa:2:5` — Question: Where does Nicole live? / Gold: `Shanghai` / raw=1, selection=0, rewrite=0
- **easy_coqa** / `coqa:339:4` — Question: Whose throw was behind Tom's by 14 inches? / Gold: `Larson` / raw=1, selection=0, rewrite=0
- **easy_coqa** / `coqa:370:6` — Question: where did Mr.s Jones live? / Gold: `down the street` / raw=1, selection=0, rewrite=0
- **easy_coqa** / `coqa:410:3` — Question: How large is it? / Gold: `4 km` / raw=0, selection=1, rewrite=0
- **easy_coqa** / `coqa:432:4` — Question: What else does he want to know how to do? / Gold: `skating` / raw=0, selection=0, rewrite=1
- **easy_coqa** / `coqa:443:5` — Question: does she normally get an allowance? / Gold: `yes` / raw=1, selection=0, rewrite=0
- **hard_convfinqa** / `convfinqa:Double_AAP/2011/page_28.pdf:3` — Question: in the same period, what was that change for the s&p500 index? / Gold: `-34.3` / raw=1, selection=1, rewrite=0
- **hard_convfinqa** / `convfinqa:Double_AMT/2010/page_34.pdf:5` — Question: considering the closing price, what was, then, their total value? / Gold: `22556579533.3` / raw=0, selection=1, rewrite=1
- **hard_convfinqa** / `convfinqa:Double_AMT/2012/page_125.pdf:4` — Question: and concerning that total cost, by how much did it change with the adjustments? / Gold: `1.7` / raw=1, selection=1, rewrite=0
- **hard_convfinqa** / `convfinqa:Double_C/2008/page_102.pdf:3` — Question: and in that last year of the period, what was the total capital? / Gold: `108.4` / raw=1, selection=0, rewrite=0
- **hard_convfinqa** / `convfinqa:Double_CDNS/2006/page_30.pdf:3` — Question: what is this s&p500 change as a portion of its value in 2001? / Gold: `11.2%` / raw=0, selection=1, rewrite=1
- **hard_convfinqa** / `convfinqa:Double_DVN/2014/page_85.pdf:3` — Question: and in the first year of that period, how much was the amortization expense for intangibles of the previous year as a po / Gold: `.8` / raw=0, selection=1, rewrite=1
- **hard_convfinqa** / `convfinqa:Double_GS/2017/page_143.pdf:3` — Question: and in that same period, what was that variation for the net comodities? / Gold: `-26` / raw=0, selection=1, rewrite=0
- **hard_convfinqa** / `convfinqa:Double_JPM/2018/page_110.pdf:3` — Question: including 2016, what then becomes this total? / Gold: `665598` / raw=1, selection=1, rewrite=0
