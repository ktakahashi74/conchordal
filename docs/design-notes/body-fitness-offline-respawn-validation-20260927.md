# 通常offline親付きrespawnの統合検証（第十七版f）

2026-09-27。隔離worktree `.worktrees/body-fitness-respawn` で、親の更新後energyによる抽選、予定子の身体候補評価、実出生、翌hopの自己除去観測を通常offline経路へ接続した。今回の検証対象は、[事前登録](body-fitness-offline-respawn-registration-20260927.md)した3 founder・単独死亡・親付きrespawn一機会である。[数値結果](body-fitness-offline-respawn-results-20260927.md)と[検査範囲](body-fitness-offline-respawn-focused-review-20260927.md)を分けて記録した。

## 固定した実装と条件

第十六版dの410 sourceをhash照合して引き継いだ。通常の初回Field出生と今回の親付きrespawnは、確定した予定子のmetadata、Voice構築、代表Recipe、72hop評価を共有する。系譜generationと身体Recipeのgenerationは別々に照合する。候補表はLog2Space全bin、peak抽選、局所探索、最終Hzの順に評価し、実子の身体・identityと照合する。出生はphonation収集後なので、そのhopの子の代謝receiptは作らない。実描画の512 samplesのゼロ自己PCMを検査し、翌hopからSourceRemoved receiptを使う。

新flag `body_fitness_respawn_offline` は既定false、offline限定、身体代謝との併用が必須。action・observation・初回Field出生の各flagとの併用や動的scenario変更を拒否する。登録sceneはHarmonic Entrain Sustain 3声、seed 7、48 kHz、512 samples/hop、380–520 Hz、背景死亡率0.5/秒、Finish 1.6秒で固定した。OFFは代謝・respawnをともに無効、ONはともに有効であり、二つの効果の単独比較ではない。Harmonic等の数値条件はsceneの固定条件であり、validatorがすべての身体へ課す制約ではない。

最終[source manifest](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-respawn/target/integration-source-20260927-v17f/manifest.json)は414ファイル、SHA-256 `22791e93aac37af82c606877d4fb5f17f68079e8fcb223aa46192cc128a58ae9`。[source archive](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-respawn/target/integration-source-20260927-v17f/source.tar.gz)も保存した。取得ごとのsnapshotはsrcのRustファイル、Cargo.toml/lockと当該integration testの248ファイルであり、全manifestも別に添付している。

## 最終検証

全suiteは1322成功・0失敗・48 ignore、`cargo test exit=0 @ 2026-09-27T17:05:17+09:00`。stdout/stderrは[test_report.txt](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-respawn/test_report.txt)、同shellで記録した実終了コードは[test_status.txt](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-respawn/test_status.txt)に保存した。fmt、標準Clippy、全target checkもexit0。通常cleanup親RNG診断の追加回帰試験は1成功（内部でRandom・Hereditary・PeakBiasedを検査）。先行v17dの7件のfocused試験も最終全suiteに含まれる。

一次通常renderはOFF/ON各2回と高閾値拒否1回、計5回すべてexit0。scene/config事前拒否は13条件を検査した。最終保存binary SHA-256は `2fd7a5fb59486d791f4501074b792a60138ab1593567924f26682bce98273cec`。frame143の最初の死亡から子の翌frame144 receiptを取得し、ONと拒否対照のFinish実dispatchはframe150/sample76800で終了した。通常respawnのintegration2件も最終全suiteで成功した。

最終版の[機械可読対応表](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-respawn/target/respawn-validation/validation.json)はsource、保存binary、実行command、scene/config、raw、各検証ログ、失敗履歴、関連文書をhashで対応づける。[一次取得索引](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-respawn/target/respawn-validation/primary-acquisition.json)から通常render正例と拒否対照へ辿れる。performance専有取得とrelease buildは今回実施していない。

## 修正と失敗履歴

| 版 | 観測・修正 | 保存先 |
| --- | --- | --- |
| 初期型確認 | source413ファイルの段階でlib check通過。未使用import警告を除去。数値取得ではない | `target/respawn-validation/preliminary-typecheck/` |
| v17 | fmtと全target check通過。Clippyは引数数3件、Cloneとの?Sized併用2件、Optionの不要なas_deref2件で失敗 | `target/integration-source-20260927-v17/` |
| v17b | 上記lintを修正しClippy通過。focused libは2成功・1失敗。試験が全bin44件と局所評価を含む65 entryを混同 | `target/integration-source-20260927-v17b/` |
| v17c | 試験を全bin Hzの一意存在の照合に修正。focused lib3件、通常render integration2件成功。scene・実装数値条件は維持 | `target/integration-source-20260927-v17c/` |
| v17d | fault試験4件と実Finish dispatch診断を追加。focused lib7件、全suite1321成功・0失敗・48 ignore。2026-09-27 16:53:14 JST、exit0 | `target/integration-source-20260927-v17d/` |
| v17e | レビューで非PeakBiasedの親RNG診断に未取得値0を報告する不備を発見。生成をPeakBiasedに限定。新回帰試験はRandomもparent_idを持つと誤期待し1失敗 | `target/integration-source-20260927-v17e/` |
| v17f | Randomの既存仕様parent_id=Noneへ試験期待だけ修正。通常cleanupを通るRandom・Hereditary・PeakBiasedの3条件で親poolと診断有無を検証。focused1件成功 | `target/integration-source-20260927-v17f/` |

v17eの報告修正は乱数列と選択計算を変更しない。以後の登録sceneの条件、閾値、modeを変更していない。各失敗は後版の成功で上書きせず、source capsuleと対応logに残した。RNG-only事前予測もコード・raw・コンパイラと依存物hashを保持する。

## 判定の範囲

通常rendererでの一機会、独立RNG選択再計算、別環境fixtureの独立Tone積分、production境界への局所fault注入は別の証拠である。記録済peakリストの重み・選択は再計算するが、通常renderのpeak抽出自体や全候補scoreを別実装で再現したとは扱わない。assert/expectによる実行中の故障検出は、scene/configの事前拒否とは異なる。

peak抽出の独立再実装と、検査範囲表に残るruntimeへの未注入faultは未確認であり、登録した全境界の独立検証完了とは扱わない。拡張前に残余項目を照合する。

次の拡張案は[反復respawnの設計案](body-fitness-repeated-respawn-draft-20260927.md)で二度目の単独死亡とsource交代を扱う。音色遺伝、同ID再利用、通常action/observationとの同時接続、非同期代謝、長期生態、実device・作者受入は未完了。第十五版bの専有計時結果はその版の証拠であり、第十七版fの実時間受入へ転用しない。F4/F5/F6全体の完了ではない。mainのsrc・既定設定・A4基準版は維持し、commit・pushは行っていない。
