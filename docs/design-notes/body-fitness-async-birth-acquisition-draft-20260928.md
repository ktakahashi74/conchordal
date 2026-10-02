# 非同期出生: 通常配送の取得案

2026-09-28。以下は取得前の入力草案を保存した記録。旧36件の取得・陰性判定は隔離版 `body-fitness-async-birth/docs/design-notes/body-fitness-async-birth-results-20260928.md`、受理修正後の2件診断は `body-fitness-async-birth-receive/docs/design-notes/body-fitness-async-birth-receive-results-20260928.md` を参照する。実装契約は隔離版
`body-fitness-async-birth/docs/design-notes/body-fitness-async-birth-registration-20260928.md`
を参照する。以下を入力・実行器・独立checkerの同じ版へ固定してから取得する。
通常の非同期配送と、配送時点を制御する部品・配線検査を混同しない。

## 入力と分母

入力生成器は隔離版の `tests/generate_body_fitness_async_birth_inputs.py`。
初版入力は `target/async-birth-input-draft-v1/` に保存した。取得前改訂のv2ではR系の物理source上限を4へ訂正した。初期foundersは3声だが、死者のtailと新子が重なる場合を含める。27本のRhai／TOMLはv1とbyte同一で、旧版は保持する。
9 scene、出生ON/OFF、a/bの36本。a/bは配送の反復観測であり、独立の生態標本とは数えない。
各sceneはseed 7、48 kHz、hop 512、30秒でFinish、評価窓は2813 hop。
同scene内はOFF-a、ON-a、OFF-b、ON-bの順。全sceneを固定順に実行し、
数値・期限・資源の失敗でも後続を取得する。固定入力・binaryの不一致は取得を止める。

| scene | 初期配置と機会 | 通常出生で確認するもの |
| --- | --- | --- |
| E-M | 空の環境でModalをField density配置、220–880 Hz | zero-mass fallbackと実出生。非平坦地形の正例には数えない |
| I-S / I-H / I-M | Sine 440 Hzの発音から1秒後、指定身体の子1声をField density配置、220–880 Hz | 最新の非平坦Cでの採点、最終jitter Hzの追加密度、実子と自己音声 |
| I-4 | 同じanchorと、1秒後のModal Field density batch 3声、220–880 Hz | 4 sourceの容量、member順、実占有を反映したspacing |
| I-W4 | I-4のField範囲を55–8000 Hzへ拡張 | 全解析帯域の準備時間と4 source経路。狭域正例だけで一般化しない |
| R-S / R-H / R-M | 同一PopulationのHarmonic・短寿命Sine・Modalを220–880 Hzに線形配置。子は指定身体の固定非遺伝template | 実枯渇後の予約、最新親energyからの抽選、PeakBiasedの非ゼロ局所探索、実子と次hopの観測 |

全Voiceはpitchをanchorで固定し、同期用beat carrierは置かない。
振幅0.06、attack cost/recharge・dissonance penalty・move costは0、
通常親と子のenduranceは100、recoveryは30。R系の短命Sineだけendurance 0.5。
background deathは0。R系は出生ON/OFFの両方で非同期身体代謝をONとし、
通常配送での身体評価から更新energy、親抽選への経路を調べる。
R系以外の非同期代謝はOFF、action身体評価は全条件OFF。

初版9 sceneは新しいdebug instrumentの`--compile-only`で全件通過し、
I系のSpawnは0秒と1秒、全sceneのFinishは30秒となった。
これは構文・IR生成の検査であり、出生成功や性能の取得ではない。

## 判定を分ける

出生機会には成功だけでなく、予約拒否、queue待ち、unsupported、取消、容量拒否、
局所cache超過、Finish時の右打切りをすべて含める。
固定ticket、各job serial、child IDから機会と全追加jobを連結し、
取消済みjobの遅延完了を別の出生へ流用しないことを検算する。

通常正例では初期Fieldの全対象子とR系の少なくとも最初の子について、
Finishより前の実consume、Recipe一致、同hopのゼロ自己PCM、
次hopの同ID・generation・birth_sampleを持つSourceRemoved receiptを要求する。
次hopで未着ならavailable=falseを失敗として残し、決定的配送の結果で埋めない。
30秒内に供給できなかった広域条件を、事後に範囲縮小・期間延長して合格へ置き換えない。
別条件として改訂取得する場合は元の失敗を保全する。

I系の非平坦性と候補間score差、density選択のmass、spacing、最終Hzの
代表Recipeと密度の対応を独立に検算する。最新Cのsupport endは判断時刻以下、
年齢は登録した4800 sample以内とする。環境epochはC再計算完了時の版を使う。
当該過去分析窓に物理的に寄与したsource集合の完全同定は現記録にないため、
現在のVoice集合をその代用にしない。

R系では同consumeのparent pool、total energy、threshold drawまたはfallback index、
選択親を検算する。各親の直前の実energy更新記録へ結び、身体評価が使われた回数と
点fallbackの回数を分ける。非同期代謝がONという設定だけで身体energyの消費と呼ばない。
親が不在となった場合や親energyが正例条件を満たさない場合も取得結果へ残す。
出生ON/OFFは出生時刻と後続環境を変えるので、WAV同一性や一般的な選択優位は要求しない。
旧版とのOFF不変対照は同じrelease profile・同一sceneで別に取る。この対照は全9 sceneで代謝・actionの身体評価もOFFとし、R系の正式取得条件とは区別する。旧新configは旧版に存在しない出生OFFキーの有無だけを違え、非同期配送の時刻差を音声bit同値の条件へ持ち込まない。

## 実行器と資源

旧paced実行器は938 hop・全hop固定Voice数を前提としていた。
旧Assayの判定は維持し、新しい出生Assayで2813 hop・可変Voice集合を記録する。
入力から最大source数を固定し、実際の全hopのID集合と容量を検査する。
音声deviceは使わず、通常wire_runtimeと512 sampleの模擬consumerを用いる。
WAV経路をinstrumentへ追加しない。

正式取得は他のcargo、全検査、render、費用測定を停止した専有枠で行う。
既存資源samplerの0.1秒採取と同iteration wait4による終了重複の契約を再利用する。
hop budget超過0、underflow0、event drop0、最大4予約、単一密度worker、
入力・出力各1、局所cache256、全bin2048を確認する。
資源上限は従来の同種非同期比較を基準として、ON/OFF最大RSS増分256 MiB以下、
ON絶対最大1 GiB以下、CPU増分1.5 core以下を候補とする。
これらの上限、登録する期限、checkerの反例は実行前に確定する。

全Rust検査・fmt・Clippy・all-targets check後のsource、binary、scene、config、
実行順、取得器、checkerをhash固定する。現時点は正式harnessとcheckerが未完成であり、
この草案だけで取得開始や受入成立とはしない。
