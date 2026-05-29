# Large-Scale BBO-Only Microstructure Validation

## Scope

This evaluates BBO microstructure alerts on Polymarket soccer moneyline markets directly. The target is future BBO repricing, not confirmed goals.

- Date range: `2026-05-06` to `2026-05-29`
- Candidate events evaluated: 17
- Events with BBO snapshots: 16

## Coverage

| event_slug | title | event_date | start_time | market_id | condition_id | token_id | market_slug | market_title | volume | status | n_snapshots | n_seconds | n_moves |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| epl-bou-mac-2026-05-19 | AFC Bournemouth vs. Manchester City FC | 2026-05-19 | 2026-05-19T18:30:00Z | 2170973 | 0x053f95738ace6c99e1571e0461feed7392fb646efa0d37ec775237e14152ecc6 | 8199493545872276685007649098855627956832512958032108818142031783023965647133 | epl-bou-mac-2026-05-19-mac | Will Manchester City FC win on 2026-05-19? | 13065430.291 | ok | 2908 | 3057.000 | 3 |
| lal-bar-bet-2026-05-17 | FC Barcelona vs. Real Betis Balompié | 2026-05-17 | 2026-05-17T19:15:00Z | 2172312 | 0x3df818c271c0c8f83f468d90c8ac2eded35bf6d1755fc17921e9116dc350c046 | 16165139802553989036572928318859780499544844336272909586947172522019046656849 | lal-bar-bet-2026-05-17-bar | Will FC Barcelona win on 2026-05-17? | 4324453.724 | ok | 2987 | 3081.000 | 3 |
| epl-che-tot-2026-05-19 | Chelsea FC vs. Tottenham Hotspur FC | 2026-05-19 | 2026-05-19T19:15:00Z | 2170978 | 0xb99047b9bdf8132c367cf327936676a1186a6eb57b86a742fdfe812f86b2e3b9 | 101407360275578242025559123794924815546227269981219987932032932255334246314286 | epl-che-tot-2026-05-19-che | Will Chelsea FC win on 2026-05-19? | 3170439.747 | ok | 2947 | 3114.000 | 5 |
| lal-ovi-ala-2026-05-17 | Real Oviedo vs. Deportivo Alavés | 2026-05-17 | 2026-05-17T17:00:00Z | 2172307 | 0x0a6ad751b54525e8b3db582c4bdeeea36dd9ea7c631948f490ed0a897beaba32 | 102348525065016859996813911607548063634346993816761209408929798820459139692065 | lal-ovi-ala-2026-05-17-ala | Will Deportivo Alavés win on 2026-05-17? | 1579001.563 | ok | 2976 | 3058.000 | 0 |
| lal-sev-rea-2026-05-17 | Sevilla FC vs. Real Madrid CF | 2026-05-17 | 2026-05-17T17:00:00Z | 2172338 | 0x0ebd0580de66c586524b1171c971cbc2459f09a7fd22e8885be2f206a1896aae | 35467927101060151903012851805640739619881978880086585496379505502168910417820 | lal-sev-rea-2026-05-17-rea | Will Real Madrid CF win on 2026-05-17? | 1492248.956 | ok | 2990 | 3042.000 | 2 |
| lal-mad-gir-2026-05-17 | Club Atlético de Madrid vs. Girona FC | 2026-05-17 | 2026-05-17T17:00:00Z | 2172309 | 0x7e7665d0024c05e1a49a71a388192d140adf29b79c09cb845603584a1df955c4 | 44664190615895681270150366705874167233547251438221554494706603007711850728283 | lal-mad-gir-2026-05-17-mad | Will Club Atlético de Madrid win on 2026-05-17? | 665464.023 | ok | 2972 | 3054.000 | 2 |
| lal-ray-vil-2026-05-17 | Rayo Vallecano de Madrid vs. Villarreal CF | 2026-05-17 | 2026-05-17T17:00:00Z | 2172317 | 0x77c178be03c194a08810e5e0798107d08d6f0abfed6a3770f7a3a0b2c9ca8622 | 73687373341669991041149908069908950294419611784418027594320727727688653746206 | lal-ray-vil-2026-05-17-ray | Will Rayo Vallecano de Madrid win on 2026-05-17? | 446494.516 | ok | 2981 | 3071.000 | 2 |
| lal-bil-cel-2026-05-17 | Athletic Club vs. RC Celta de Vigo | 2026-05-17 | 2026-05-17T17:00:00Z | 2172333 | 0x8db55aa3e852ecdc19750562192ba86bae8b5ef5aeffa2fadb612ca931dbf93d | 43776481480363064542433907779951959910657837388360544485428342944772738825988 | lal-bil-cel-2026-05-17-bil | Will Athletic Club win on 2026-05-17? | 263724.310 | ok | 2980 | 3032.000 | 2 |
| lal-elc-get-2026-05-17 | Elche CF vs. Getafe CF | 2026-05-17 | 2026-05-17T17:00:00Z | 2172339 | 0xd393060ceda70c38894cf32b183d1a654dbefaab72212cb3639d877f5d09652f | 94487636525931339299021229942080704713884918804218975451237855248167550613083 | lal-elc-get-2026-05-17-elc | Will Elche CF win on 2026-05-17? | 199813.214 | ok | 2986 | 3083.000 | 3 |
| lal-lev-mal-2026-05-17 | Levante UD vs. RCD Mallorca | 2026-05-17 | 2026-05-17T17:00:00Z | 2172302 | 0x9005cf9b22db2d7bb9fccf9443e2c41e501cb8bc7544abe3f152e8325839847b | 93693604903771587115434052616281707898754720600760426782652710262423522512790 | lal-lev-mal-2026-05-17-lev | Will Levante UD win on 2026-05-17? | 194706.683 | ok | 2954 | 3139.000 | 3 |
| lal-rso-val-2026-05-17 | Real Sociedad de Fútbol vs. Valencia CF | 2026-05-17 | 2026-05-17T17:00:00Z | 2172349 | 0xfbf7591d30925399ba090c02ef454b7f6500815c1defe92b4700e295fdb3be80 | 114205872873319726218128572778917530104087936394132426052866644318187998251316 | lal-rso-val-2026-05-17-val | Will Valencia CF win on 2026-05-17? | 149184.409 | ok | 2980 | 3018.000 | 3 |
| lal-osa-esp-2026-05-17 | CA Osasuna vs. RCD Espanyol de Barcelona | 2026-05-17 | 2026-05-17T17:00:00Z | 2172325 | 0x60b499a898f654c92cd00ff248e085f29be154b4525dbf130aeb43ef11013aa4 | 52113012205596044343788817264242022664675604496949762437917318878623675153882 | lal-osa-esp-2026-05-17-osa | Will CA Osasuna win on 2026-05-17? | 112995.435 | ok | 2969 | 3056.000 | 3 |
| egy1-ems-eas-2026-05-20 | El Masry SC vs. El Ahly SC | 2026-05-20 | 2026-05-20T17:00:00Z | 2176984 | 0xdb34274928ee8994b83e553e34f3f3218d9365084ca58298d03b9e7944dfc3a5 | 113575008495194320802189032073568910876727627723225547391905904406544750821315 | egy1-ems-eas-2026-05-20-eas | Will El Ahly SC win on 2026-05-20? | 85048.815 | ok | 2965 | 3096.000 | 5 |
| fif-mex-gha-2026-05-22 | Mexico vs. Ghana | 2026-05-22 | 2026-05-23T02:00:00Z | 2181609 | 0xd64eb22f3cded78a9758ce416cee2de2398946b5bb03654f400ad308127abf39 | 85710261221715064185161469213775803122690034661074221789837511032607262080610 | fif-mex-gha-2026-05-22-mex | Will Mexico win on 2026-05-22? | 72783.661 | ok | 2915 | 3310.000 | 7 |
| egy1-zas-ccc-2026-05-20 | Zamalek SC vs. Ceramica Cleopatra Club | 2026-05-20 | 2026-05-20T17:00:00Z | 2176979 | 0xa34cce7684e6f1c8d7a2325ce9a2f6bd5577662102228280de34c72ecb244f64 | 89796353025645847117291334665679952842982522290797953815896118583927120511304 | egy1-zas-ccc-2026-05-20-zas | Will Zamalek SC win on 2026-05-20? | 60863.447 | ok | 2951 | 3056.000 | 2 |
| egy1-pyf-sms-2026-05-20 | Pyramids FC vs. Smouha SC | 2026-05-20 | 2026-05-20T17:00:00Z | 2176977 | 0xf596b66e195a0f8537eb555d952f92512d797432b7e4c1061cf3ff010dc6630d | 8113369331580148760672209006672381990610649646199244145355687323782725704362 | egy1-pyf-sms-2026-05-20-draw | Will Pyramids FC vs. Smouha SC end in a draw? | 30068.662 | ok | 2957 | 3086.000 | 3 |
| ukr1-sp-dyn-2026-05-16 | SK Poltava vs. FK Dynamo Kyiv | 2026-05-16 | 2026-05-16T10:00:00Z | 2188334 | 0x18157ea18f904369186548ea60ae93dc11192d197b14886074e9904539be7481 | 94165639410114693120418822813972237815699455260990603925224930772333580084945 | ukr1-sp-dyn-2026-05-16-dyn | Will FK Dynamo Kyiv win on 2026-05-16? | 24359.787 | no_bbo | 0 |  | 0 |

## Aggregated Strategy Scores

| strategy | n_events | n_alerts | true_alerts | false_alerts | covered_moves | total_moves | median_lead_sec | precision | move_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mid1c | 15 | 73 | 5 | 68 | 5 | 48 | 5.000 | 0.068 | 0.104 |
| mid1c_and_spread2c | 15 | 28 | 1 | 27 | 1 | 48 | 4.000 | 0.036 | 0.021 |
| mid1c_and_update10 | 15 | 70 | 5 | 65 | 5 | 48 | 5.000 | 0.071 | 0.104 |
| spread2c_or_depth50_update10 | 15 | 117 | 6 | 111 | 6 | 48 | 9.000 | 0.051 | 0.125 |

## Interpretation

This is the correct larger-scale test for BBO microstructure: predict BBO repricing itself across many markets. To connect back to LSports goals, we still need more LSports fixture mappings for these Polymarket markets.

Figure:

- `outputs/figures/large_scale_bbo_micro_strategy.png`
