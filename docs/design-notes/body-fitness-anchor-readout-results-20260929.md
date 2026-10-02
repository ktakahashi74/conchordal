# 部分音群の採点位置：単一介入診断の結果

2026-09-29。[固定した診断契約](body-fitness-anchor-readout-diagnostic-20260929.md)に従い、群・power・motionを共有したまま、採点位置だけをERB重心から選択済みanchorへ変更した。代表描画を正本とする境界と既存許容差は変更していない。本番runtimeには接続していない。

## 結論

**採点位置の変更だけでは描画参照への精度条件を満たさない。** 誤差上限を超える候補は159件から93件へ減ったが、最大誤差と明確な順位逆転は増えた。この結果からanchor版を採用せず、資源の再測定にも進まない。

| 指標 | 凍結v2 | anchor介入 | 固定条件 |
|---|---:|---:|---:|
| 誤差上限超過候補 | 159 / 728 | 93 / 728 | 0件 |
| score最大絶対差 | 0.3971686363 | 0.5726536512 | 0.025以下 |
| level最大絶対差 | 0.1268229783 | 0.2675193250 | 0.0125以下 |
| 旧score差0.1以上の厳密な順位逆転 | 3対 | 4対 | 0対 |

v2から85候補が回復し、19候補が新たに不合格となった。介入後の順位判定は104組、対象345対で、新たな同点化は0対。全728候補でproducer再生とcontrolの直接列挙が先行照合を通過し、欠測・unsupported・解釈不能候補は0件だった。

旧v2のscore最大例（sine_440環境、harmonic_dark、基音440 Hz、0 cent）の誤差は0.3971686363から0.0751783252へ減ったが、上限0.025をなお超える。介入後のscore・level最大例は同環境のharmonic_spread、基音440 Hzから24 cent上、候補446.1421814 Hz。v2のscore誤差0.0564141423に対し、介入後は0.5726536512となった。失敗件数の減少だけを全体の精度改善と呼ばない。

## 実装と検算の範囲

未完状態は先に `.worktrees/body-fitness-direct-model` の新規ブランチ `body-fitness-direct-model`、commit `0f0f4f8d9f7e08091af08e726da2bb4f302bb76b` へ原状退避した。そこから診断moduleの集計と独立checkerだけを修正した。既存のv2群形成・身体準備・凍結入力は変更していない。

修正は、先行照合を通らない候補の解釈集計からの除外、候補・群別・総summaryの独立照合、必須raw bin massの検査、指数・参照power・sigmoid係数・frame数・body・入力hashの固定条件との照合である。診断値の保存と、解釈可能な値の分母を分けた。

[全取得](../../target/body-fitness-support-20260929/anchor-v1.jsonl)は環境4行・候補728行・順位104行・case集計13行・総集計1行の850行。取得JSONLのSHA-256は `8c0f5b7ba5c9a2ed7dac0ff3c51b92c323805f0e5d95a34b81e0a6906d5a4bdc`。取得用ignored testのexit 101は精度条件の不合格を表し、途中打切りではない。

[独立checker](../../target/body-fitness-support-20260929/readback_anchor.py)は保存された群とmotionから両経路のbin massを再構成し、候補値と全集計を照合した。[検算結果](../../target/body-fitness-support-20260929/anchor-v1-independent.json)は内部照合成功、旧描画との精度条件不合格（exit 1）。[主担当の別集計](../../target/body-fitness-support-20260929/coordinator-reduction.json)でも保存score・levelの誤差、回復・新失敗、順位を再計算し、producer集計との一致を確認した。これは群形成やPCM/NSGTを全728件で独立再生成したという意味ではない。

対象unit4件、fmt、標準Clippy、all-targets checkは成功した。独立checkerへsummary欠落・値改変、raw欠落、body・hash・metadata・先行照合flagの単独変異を与えた反例7件は、すべて内部検算エラー（exit 2）として拒否された。初回反例fixtureには浅いcopyによる変異累積があり、その記録を保存した上で、元入力のdeep copyから各変異を独立生成して確認し直した。数値取得は再実行していない。

全Rust回帰は初回・再実行とも1,315成功・1失敗・61 ignore、exit 101だった。`changed_body_cancels_old_job_without_advancing_pitch_then_recovers` が非同期準備の30秒待機で失敗し、該当検査の単独再実行は成功した。この2回の該当ソースは退避commitと同一だったが、それ自体は全体回帰の合格根拠ではない。

続いて対象requestだけの `cfg(test)` 進捗記録を追加し、既定並列の全suiteを一回観測した。30秒・assert・132候補・並列選択規則は変更していない。結果は同じ1,315成功・1失敗・61 ignoreだった。[記録付きログ](../../target/body-fitness-support-20260929/full-test-trace.log)では旧job取消回収後、新job serial 2が2 laneで候補を計算中であり、左56/66件・右59/66件完了時点で期限に達した。pendingは残り、readyはなく、join・応答送信・Action受信には未到達だった。対象単独の記録付き試験は2.75秒で成功した。失敗区間は新jobの候補計算に絞れたが、CPU競合等の根本原因や計装なしの時間は確定しない。記録自体にも負荷がある。

単独成功を全体回帰成功とは扱わない。[失敗監査](../../target/body-fitness-support-20260929/full-test-failure-audit.json)、[進捗監査](../../target/body-fitness-support-20260929/full-test-trace-audit.json)、3回の全suiteログ、worktreeの `test_status.txt` を保存し、診断修正と進捗記録は未コミットのまま保持する。観測後に記録用indexと試験/本番のsend分岐だけを整理し、最終sourceでは単独検査・fmt・標準Clippy・all-targets checkが成功した。全suiteは追加反復しておらず、その観測版と最終計装版を分ける。数値取得時のsource・binary・入力hashと検証状態は[manifest](../../target/body-fitness-support-20260929/anchor-acquisition-manifest.json)に保存した。後続の試験用計装を数値取得時のsourceと混同しない。

## 残る境界

旧描画参照は選択peakのbinへ主観強度を置く。本介入は解析器のpeak検出そのものを再現せず、部分音群が保持するanchor Hzと補間を使う。再配分後重心との違いを一つ取り除いても、両者の同値性は成立しない。位相干渉、解析窓、時間集約、名目帯域外laneの扱い等もこの介入では変更していない。今回の失敗全体をそのいずれか一つの原因へ帰属しない。

入力は設計に使用済みの保存728候補で、holdoutではない。凍結入力のLandscapePeaksは実Harmonicのまま保持した。通常runtimeの抽選・占有・親選択、同hop出生、全hop費用、実device、作者受入は未検証。v2の資源10/20成功という旧判定も保持する。

次の技術単位を作る場合は、今回の残差と描画参照の採点位置・質量配分を区別し、代表描画に照合する近道の契約へ戻る。閾値緩和、対象負荷の縮小、同じ条件の再測定で採用へ進めない。音色Task 2の自己モデル面との共同設計と、I12b終結・基準版固定 → I4配線 → body-aware F3 → 音色Phase 3の統合順序は維持する。
