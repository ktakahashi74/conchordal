# F4e PeakBiased 出生一機会の結果（2026-09-27）

事前登録: `body-aware-fitness-f4e-registration-20260927.md`。対象は parent 不在の PeakBiased 一回の死亡・補充機会。局所探索半径は 0.1 半音、step は 0.1 半音。通常 runtime 配線は含まない。

実 habitat の解析結果と実 candidate body から、380–520 Hz の全 44 Log2 bin を評価した。body score scan から 11 peak を抽出し、それらの局所探索 grid 32 点の実 body/recipe も評価した。点 score scan を一様化した場合の peak 候補集合とは異なる。選択の独立 `WeightedIndex` 参照と RNG probe は一致。選択周波数は 440 Hz（bit pattern `1138491392`）。低閾値 `0.38962239027023315` では同じ body/recipe の子を一人生成、高閾値 `0.8896223902702332` では生成ゼロ。どちらも次 hop の重複補充なし。子 ID、解析 epoch、template、局所探索点の recipe を古く／異なる値にした四例は、選択前に拒否した。

局所探索は非ゼロの grid を実行し、各点の body score を参照した。この機会では中心 440 Hz が最大だったため、最終周波数は中心から移動していない。parent ありの重み・世代継承、初回 Spawn、通常 runtime observer の供給と費用は未検証。オフライン `cfg(test)` の表に候補の実 body/fitness を載せた境界のみ通過。

取得物: `target/f4e-offline-evidence/f4e-result.json`。取得前 source SHA-256 は `respawn.rs=1ec6a485f626c77535965088913b2e13fd9c3f36f0e1aaec7e6578dd8cbfde88`、`f4b_offline.rs=5d9681cbf35fa2aa087caa0891bb1d9fd60ad63293f86b104b916460bf0c19a1`。取得後は `respawn.rs=38b90852825fdb21cae55bbb2e40db29799a15a834c02f7f941dfda7e1668e4a`、`f4b_offline.rs=35ced18de3700bb9fd8a64c755cacf5b09f18a7d80330f3de94db9d6e509a381`。結果 JSON SHA-256 は `50dc22bb7330a17c5565e39942e55d3e99213b823832b8ce9e81169a4246fd1c`。

Focused test 通過。`cargo fmt --all` と標準 `cargo clippy -- -D warnings` 通過。全 `cargo test -- --nocapture` も exit 0（2026-09-27 12:10:14 JST）。stdout/stderr は `test_report.txt`、同一 shell で得た終了コードは `test_status.txt`。ログ SHA-256 はそれぞれ `0970db10970dc5b12896586e5aa816d21be96acfaeed11f89b85ee30b617323d`、`6be4e02b54327ce7cd7b193b46e78d6b3b27eb017dffdc6cbdc4bfacaead8e46`。
