# 通常offline出生と代謝の結合: 第十六版dの結果

2026-09-27。対象は[事前登録](body-fitness-birth-metabolism-registration-20260927.md)に固定した、通常の `conchordal-render` による初回Field出生と身体代謝の併用である。seed 7、48 kHz、512 samples/hop、Sine 440 Hzの環境Voice、Harmonic Entrainの子、`consonance(380, 520).peak().spacing(0)`、出生後0.4秒、比較閾値 `1e-6` を取得後に変更していない。出生だけONと出生・代謝ともONを各2回実行した。

## 取得と出所

一次記録は全suite内の結合試験が生成した[summary](/home/shafi/lwrk/conchordal/target/body-fitness-birth-metabolism-build-20260927/runtime-birth-metabolism-evidence/registered-4162-1790492330438638435/summary.json)と[rawディレクトリ](/home/shafi/lwrk/conchordal/target/body-fitness-birth-metabolism-build-20260927/runtime-birth-metabolism-evidence/registered-4162-1790492330438638435/)である。各modeの2回分について、シナリオ・config・WAV・JSONL report・stdout/stderr・取得時のbinary hashを保存した。[取得対応表](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-birth-metabolism/target/birth-metabolism-validation/primary-acquisition.json)は一次記録と先行focused記録を結ぶ。先行focusedの[summary](/home/shafi/lwrk/conchordal/target/body-fitness-birth-metabolism-build-20260927/runtime-birth-metabolism-evidence/registered-168-1790492186726247936/summary.json)は一次記録と一致した補助照合であり、一次記録の代用ではない。

第十六版dの[source manifest](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-birth-metabolism/target/integration-source-20260927-v16d/manifest.json)は410ファイル、manifest自体のSHA-256は `01b8cc1c04aa9ffc2d79fc015f74852f7fb6d488215bff934a4394b8b4fddcaa`。一次取得にはこのmanifestを外部指定し、取得時のsource file hashを全件照合した。先行focused取得は自身の245ファイルのsnapshotをmanifestの対応エントリと照合したもので、focusedが410ファイルを個別に記録したという意味ではない。4回のrenderに使ったbinaryのSHA-256は `309ab30716e83bab6e1c11efcd4506330df89774706d31f62e215549cd3575d8`。[binaryコピー](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-birth-metabolism/target/birth-metabolism-validation/render-binary)も保持した。

## 観測した結合

出生決定と主解析の支持終端はともにsample `33280`、frame `65`。対象範囲内の44 binを候補として実子Bodyの代表Recipeで評価し、選択bin `288` の中心440 Hzから既存のbin内選択を経て、実子の最終基音は `440.7801818847656 Hz` となった。44候補のbin・値・Recipe identity、選択された候補と実際のSpawn周波数、最終identity（source ID `2`、出生時body generation `0`、32 byteのRecipe hash）を照合した。fallbackは `none`。これは登録された離散bin候補の評価と実出生の照合であり、連続周波数全域の直接最適化を意味しない。

出生hopの有界transition reportでは、observerの出生前batchは1 source。環境Voice `1` は `source_removed`、子 `2` は `birth_shared`。両者のdecision sampleとsupport endは `33280`、epochは `0`。子のsource identityは `(id=2, generation=0, birth_sample=33280)`、receiptのscoreは `0.7436285615`、levelは `0.8156661987`、energyは `0.9955368042`、body generationは `1`。次のframe `66` ではobserver batchが2 sourceとなり、両Voiceが `source_removed`、decision sampleとsupport endは `33792`。子のsource identityは維持され、score `0.7436284423`、level `0.8156661987`、energy `0.9910736084`、body generation `8` を記録した。後者のgeneration増加は代表Recipeの識別更新であり、8種類の物理的なBody・音色変更を測ったものではない。両sourceのspaceは690 bins、55–8000 Hz、96 bins/octで一致した。子の出生hopの `current_hz` は実最終基音と一致し、翌hopは評価時Recipeの基音として正の有限値であることを検査した。

frame `96` の子のEntrain energyは出生だけONで `0.85603476`、結合ONで `0.85717773`、差は `0.00114297`。事前閾値 `1e-6` を超え、結合ONの値は代謝reportに記録された実energyと一致した。環境SineのEntrain数は0、平均energyは `null`。各mode内の2実行はWAVと対象reportが一致した。両modeのWAV SHA-256も同じ `761da54cb628d3a427540e651d511457921c79a320f6e62b0cf314b21c91be90` だったが、mode間のWAV一致は登録上の一般条件ではない。

## 負例と局所検査の範囲

[シナリオ拒否raw](/home/shafi/lwrk/conchordal/target/body-fitness-birth-metabolism-build-20260927/runtime-birth-metabolism-evidence/scenario-gates-4162-1790492330438609495/)で、初回環境なし、時刻0のField子、同hopの複数Spawn、1 Spawn内の子2声、第三Voice、Release、途中Update、respawn、解析設定変更、action/observation同時ON、実時間instrumentを配線前に拒否した。実行前に拒否したケースではWAV・reportを生成しないことも試験した。既定flagはfalseで、通常のinstrumentへの結合経路はない。通常Rhaiが自動採番するID・populationの重複を外部シナリオとして注入した取得はない。

無音の新生児に512 samplesのゼロ自己PCMを作り、出生hopで2 sourceをobserverへ渡して次hopの完全batchを得る経路は[局所試験ログ](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-birth-metabolism/target/birth-metabolism-validation/focused-zero-pcm.log)で1件成功した。結合integrationの[focusedログ](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-birth-metabolism/target/birth-metabolism-validation/focused-coupling.log)は2件成功。全suiteは1311件成功、失敗0件、ignored 48件で、[同じshellに記録した終了コード](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-birth-metabolism/test_status.txt)は `0`（2026-09-27 16:01:11 +09:00）。[全出力](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-birth-metabolism/test_report.txt)を保持した。format、標準Clippy、全target checkも[checks](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-birth-metabolism/target/birth-metabolism-validation/checks.json)に終了コード `0` を記録した。全体検証後も第十六版dの410 source hashは不変。

出生前batchのsource欠落・余分・birth sample違い、出生失敗や誤世代、worker欠測・PCM不連続、自己PCM欠落・長さ違い、未来または期限切れのsupport、epoch/space不一致、翌hopの子欠落は、配線上assertまたは拒否する境界として実装した。ただし各故障をこの通常renderの独立fixtureへ注入した結果ではない。自己除去の数値的な独立検査は既存のcore birth-history試験、2候補の積分評価は既存の出生試験に依拠する。今回それらを新しい数値oracleで再取得したとは扱わない。

## 修正履歴と残余

失敗時のrawとログを保持した。初回第十六版ではtest呼出しの引数7箇所が合わずcompile停止。第十六版bではゼロPCM試験のacoustic clock更新漏れで局所試験失敗。第十六版cでは通常のRhai Spawnが同じeventに付ける `SetPopulationCrowdingTarget` を結合gateが拒否した。[compile-only IR](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-birth-metabolism/target/birth-metabolism-validation/registered-compile-ir.log)に、両Spawn直後の同population・1回のcrowding actionを記録した。第十六版dではこの通常形だけを受理するようgateを修正した。続くfocusedの初回失敗は、testが32 byte配列の `recipe_hash` を `u64` と誤認したassertであり、実reportの候補・identityの欠落ではなかった。assertのみ修正し、固定シナリオ、係数、閾値は維持した。[検証ログ群](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-birth-metabolism/target/birth-metabolism-validation/)と一次rawを保存している。

この結果は1環境Voiceと1子の初回Field出生、出生hopから次hopの共有環境・自己除去環境の切替、frame 96までの代謝効果に限る。親付きrespawnは未接続で、[別設計案](body-fitness-offline-respawn-draft-20260927.md)の段階。動的解析変更、長期生態、実時間の公平性・期限・計算費用も未取得。第十五版bの10秒計時を第十六版dの実時間性能合格へ転用しない。
