# F3e 第四段階: live-paced worker 取得結果

日付: 2026-09-27。状態: 隔離 worktree・固定 Sine 身体の release 試験専用取得。取得前登録は `body-aware-fitness-f3e-phase4-live-registration-20260927.md`（SHA-256 `6192332d6b52447eb076332e77d639545eb74816687aa9f58600e4e3ce32297c`）。release test binary SHA-256 は `353bbcc40b485e6776a4ecd9ef4a58cbd5040a4c6ed763dbe8aa42776ceb962a`。1 source と 4 source を各一回だけ実行し、再測定による選別なし。

| 条件 | 1 source | 4 source |
| --- | ---: | ---: |
| 予定・処理 hop | 938・938 | 938・938 |
| 予定時間 | 10.005334 秒 | 10.005334 秒 |
| deadline miss | 0 | 0 |
| source別実身体表消費 | 5 | 5, 5, 5, 5 |
| source別 Ready 機会 | 5 | 5, 5, 5, 5 |
| source別延期 gate | 933 | 933, 933, 933, 933 |
| source別窓内 q 完了 | 5 | 5, 5, 5, 5 |
| source別窓外 q 完了・join後回収 | 1・1 | 各1・各1 |
| F3b受信・失効 | 935・0 | 937・0 |
| 最大保持 全体 | 1 | 4 |
| 最大開始遅れ | 0.071781 ms | 0.070317 ms |
| 終了値 | 0 | 0 |

両条件とも source別の拒否理由は空、実score利用は1 sourceで722回、4 sourceで722／723／722／723回。実 q thread の個別所要時間と全完了時刻は生ログJSONに保存した。1 sourceの完了時刻は開始後1.708／3.440／5.164／6.876／8.589／10.299秒、個別所要時間は1.700–1.721秒。4 sourceの完了時刻は各source順に1.755–10.508、1.744–10.628、1.757–10.611、1.754–10.588秒で、個別所要時間の全体範囲は1.737–1.784秒。各 source の6番目の要求だけが10.005秒の測定窓を越え、測定終了後のjoinで回収された。これらは実身体表消費に数えない。計時中は `try_recv` のみ使用し、密度threadの `join` と残結果回収は938 hop後。F3bも計時中 blocking receiveなし。

固定 Sine 440 Hz、固定Recipe／routing／epoch、`landscape_weight=0`、`move_cost_coeff=10` の政策では、連続 gate でも実数値の身体表消費が正になり、登録した deadline miss は0だった。一方、938 gate 中933 gateを延期し、実判断は各 source 5回、約0.5回／秒。これは応答頻度を大きく下げる代償を示す。今回の live-paced 検査は `cfg(test)` 専用であり、別に統合された通常runtime観測を代替しない。通常runtimeへの延期政策の非同期配線、身体・pitch変化中の有効率、音楽的妥当性、一般的なDSP性能、公開既定の採用は示していない。

生ログ・終了値:

| 条件 | log SHA-256 | status SHA-256 |
| --- | --- | --- |
| `targeted-logs/f3e-phase4-live-1source.log`／`.status` | `c04a0905d4b929da54d4ea2ca9757fccffe8e681d4f01b398aaaf09eef68b8db` | `55429a051d222d58d9281470bf707c6b9c5c49cb2116947763d48e247474e0a4` |
| `targeted-logs/f3e-phase4-live-4source.log`／`.status` | `96748906c7aaae4e7ba497d55f7669872f3763dd7db0ad14e81a2555a0122a4d` | `9fd35a59299fff13130f25399a65b76719fc960b6b767891683d49470f26e14b` |

source・元／改訂／live登録・生ログ／終了値・全suiteログ・release試験binaryを `.worktrees/target-body-fitness/f3e-phase4-sealed-20260927-115723/` に対応付けて保存した。`SHA256SUMS` 自体の SHA-256 は `f5798129720de340de229ab2fd97762cbb6ae8e21da0528d32745f99a6926871`。この capsule は封印時点の結果文書も保持し、後から追記した本段落は元文書側に置く。
