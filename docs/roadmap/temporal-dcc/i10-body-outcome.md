# I10: 指令と私有実音の帰結診断

範囲確定（2026-09-20）: 機構選択規則（[technote-ledger §9.3.55](../../design-notes/technote-ledger.ja.md#mechanism-selection-rule)、[milestones §1.3](milestones.md#13-機構選択規則による本体範囲の縮小2026-09-20同日のレビューを反映)）により、I10は正本の狭い範囲で閉じる。共同推論・全帰結投影・phrase再重み付け・groove評定headは研究拡張へ移し、`joint/`・`consequence/`・`long_form.rs` はcfg(test)のまま凍結する。残作業・仕分け・I11入力契約は末尾の[「I10の範囲確定と残作業の再定義」](#i10の範囲確定と残作業の再定義2026-09-20)を正とする。以下の「最新状態（2026-09-18）」行と2026-09-20までの各節は履歴として保存し、現行の完了条件ではない。

最新状態（2026-09-18）: onset／release観測、身体世代付き六記述値に続き、動作条件付き自声予測を通常runtime／report／UIへ接続して検査した。活動・終端の5座標と将来の六記述値を、固定／動作非依存対照とともに発行時の予測で照合・学習する。私有参加traceの有界Rust数値filterも移植し、Python参照と照合した。続いて、共有memoryの保持状態とretrieval score上下限を通常worker／report／UIへ接続した。共有reference inventoryも原観測の2秒支持とanchorから通常worker／report／UIへ接続した。実Voiceの発行時reference／予測凍結、実onset／releaseからの私有trace更新、両対照の診断も通常rendererへ接続した。候補帰結の数値核（距離、候補混合、ordinal backoff、時刻lookup、default差分）は独立参照と照合した。実音288 recordからPAM prototypeを作り、通常身体worker／report／UIのdescriptor照合を接続・検査した。共有音群の2秒descriptor cacheとprototype対応も両busの通常worker／report／UIへ接続・検査した。局所own-excluded energyの中心補間と16点診断も通常参加reportへ接続・検査した。区間内の欠測を見落とさない支持bitsetを追加し、実音6,336条件のenergy積分監査で16点近似の誤差を記録した。続いて、指令発行時のwhole-source energy予測と16窓の私有実音教師・両対照・学習前誤差を通常renderer／report／UIへ接続した。さらに、同じ発行時点のhabitat自声除外予測とのenergy比率・区間支持・除外した予測質量を通常report／UIへ接続した。観測と候補で共用する10個の生特徴数値式も分離し、出所タグとfeature manifestのraw投影方針を追加して検査した。7 classの実身体比較素材には、512／480 hopで発行前PCMを保存するv2取得を追加し、未来PCMとともに独立検証した。さらに512-hop素材を通常の私有NSGT・raw抽出へ通し、396分岐・両busの297,000未来hopについて十生特徴と原支持を独立照合した。観測と候補で共有する物理窓集計も通常phraseへ接続し、32時点×3窓幅の76,032窓で数値・出所・支持を独立照合した。7 classの身体入力則と有界時刻生成も共通化し、取得器の396分岐が不変であることを確認した。I10全体は未完了。条件付き身体行動profileと7 classのprimitive特徴窓は通常接続・検査済み。次は評価時刻の確定、全head特徴・帰結表・actual-Voice/default対応、全I10資源検査。

2026-09-14。機構選定の5文書を `329f638` にcommitし、「commit しI10に進む」に基づき実装した。初期onset診断の通常runtime接続と検査を完了した。
今回の終了点は、固定身体・固定方策の一Voiceについて、onset指令から実音を経た診断を通常runtime、report、UIへ通すこと。
I10全体、学習した自声予測、六記述値、私有参加trace、候補帰結表、I11の生成作用はこの単位の完了に含めない。

## 実装前に固定した観測契約

- 既存の `temporal_mode("observe")` で有効にする。reportの有無では有効性・計算・記憶を変えない。offでは観測器を作らない。
- rendererへ渡された `ToneCmd::On` を指令とする。検討だけの候補は入力しない。指令受領時刻、予定onset、Voice ID・世代、tone IDを保持する。
- 重複、spec欠落、cutoff、身体の生成拒否は指令の拒否として報告する。指令の受理だけで発音成功としない。
- 実音は当該toneを実際にrenderしたsampleを、habitat／presentationのrouting後、mixture合算・guard前で測る。個体・toneの分離は合成器が持つ私有情報であり、知覚的な音源分離の証拠ではない。
- 観測窓は予定onsetから `ceil(0.020 * sample_rate)` sampleの半開区間とする。予定時刻は窓の位置だけに用い、活動開始の検出値として代入しない。
- 活動開始は実sampleの絶対値が `1e-5` full scaleを初めて超えた時刻とする。peak、RMS、最初の閾値超過を各busに記録する。この閾値と20 msは工学的検査条件であり、知覚閾値・普遍的onset・発音失敗の定義ではない。
- 全窓の観測済みsample数を分母とし、取得欠落は埋めない。完全観測のゼロ、閾値以下の非ゼロ、閾値超過、未routing、不完全観測、非有限sampleを区別する。
- toneが終了して以後の、実際にrenderされた区間の寄与は既知ゼロとする。未renderの尾部、EOF、時刻の飛びは既知ゼロにしない。記録には元窓と観測済み数、実際の利用可能時刻を残す。
- 観測pendingと各hopの出力は各256件を上限に事前確保する。上限超過は件数を報告し、発音・既存の処理を止めない。これは認知的忘却のモデルではない。
- 完了記録をreportへ排出し、同じ計算済み状態の累計と最新記録をUIへ渡す。個体の指令・分離音を共有P／ListenerTwinへ渡さず、音をinstrumentからdiskへ書かない。
- rendererのsample clockはこの私有測定の時計とする。NSGTのepochと同一視しない。新しいruntimeは空状態から開始し、Voice世代は指令時に凍結する。

## 検査対象

実sampleによる遅延検出、routing隔離、既知無音と欠測、指令拒否、上限、tone退役後の窓、世代・slot再利用、EOFの不完全窓を検査する。
通常renderでは固定seedのonsetを用い、off／observeとreport有無の音声一致、reportの指令・実音の時刻と支持を比較する。
認知機構の学習や音楽的な有効性の比較とは区別する。

## 接続した動作

`PhonationBatch` にVoiceの世代を載せ、指令時点で凍結する。
`ScheduleRenderer` は受理・拒否したOn指令を `life::action_observation::Observer` へ渡す。
観測中のtoneだけが、実際に合成したsampleで各busの記録を更新する。toneの終了後も、renderされた区間の既知ゼロを窓へ含める。
元の合成順序・routing・乱数を維持し、別のtoneやVoiceの音を当該指令の帰結に加算しない。

workerはobserve時だけ観測器を構築する。各hopでreportの有無に関係なく出力を排出するため、
診断の有効性・pending・累計はreportへ依存しない。reportは `self_sound_outcome` と終了時の
`self_sound_observation`、UIは同じ観測器の累計と最新記録を読む。UIの表示はListenerTwinと区別した
「Voice self-sound observation」に置いた。EOFで残った窓は追加無音を作らず、不完全観測として閉じる。

pending・出力bufferを各256件で事前確保し、通常の更新・排出で再確保しない。
既存toneには観測slotを保持する。UI snapshotのcopyとreport処理は既存の各経路の費用へ加わる。
全workerのallocation数や実機でのheadroomは今回測定していない。

## 検査結果

全Rustテスト1,014成功、失敗0、30ignore。
`cargo test exit=0 @ 2026-09-14T09:04:10+09:00`。全出力と同一shellの終了コードを
`test_report.txt`／`test_status.txt`へ保存した。Clippy、全target check、format、diff検査も通過した。

- 観測器の4検査で、指令より遅れた実sampleの活動検出、独立計算したRMS、既知無音／閾値以下／欠測／EOF／非有限値を照合した。
  世代を持つ新規指令、旧slotの別sourceによる再使用、逆順inputの非加算、routing、拒否、pending／出力の上限とbuffer位置の維持も検査した。
- rendererの検査ではsine／harmonic／modalの3身体と4 routing条件を使い、別個体が同時に鳴る混合音と、
  対象個体だけを独立にrenderした音を比較した。私有RMS・最初の活動sampleが独立参照と一致し、観測on/offの両bus音声も一致した。
  指定onsetと検出時刻は一致する場合もあるため、一律に遅延するという仮定は置いていない。
- 指令重複、spec欠落、cutoff、生成拒否、未完の観測窓をrenderer経路で検査した。拒否を発音済みとして報告しない。
- 既存の `temporal_observation_preserves_audio_and_finishes_both_buses` を拡張し、通常binaryのoff／observe・report有無で
  音声の完全一致と、presentationだけの実音、Voice ID・世代、指令時刻／検出時刻／利用可能時刻、960-sampleの支持を確認した。
  同じ検査の他の認知診断設定とも併用でき、終了時pendingと容量・出力dropは0だった。

これらは合成器内部の指令―実音測定と非干渉の証拠であり、聴覚的なonset検出精度、予測学習、実機再生、可聴性・作者採用の証拠ではない。
今回の実装は未commitで保持する。

## 残るI10

2026-09-14時点の次工程はrelease観測だった。2026-09-17に下記の接続・検査を実施した。
六記述値と身体model世代、発行済み自声予測と実音の照合、固定モデル／動作非依存モデルの比較は下記の通常経路へ接続した。
参加時刻の私有trace・credit、候補帰結表、I10全体の資源検査は未完了。自声予測の実装だけでこれらを代用しない。
I11の生成作用は無効のまま、該当R4/R5・A1/A2の判定を維持する。

## I10継続の実装契約（2026-09-17、実装前登録）

I10全体の終了条件は維持し、release観測、自声の六記述値と身体世代、発行済み予測の照合、
私有参加trace、候補帰結の数値経路、両busの診断、資源検査を順に接続する。I11の生成作用は有効にしない。

最初の追加単位はrelease指令の帰結観測とする。予定releaseから2秒の半開窓を使い、
実際のsampleの最初・最後の活動、peak、RMS、観測済み支持と、rendererで確認したtone終端を記録する。
活動閾値は既存の1e-5 full scaleを維持する。最後の閾値超過は知覚的終端ではなく、
rendererの終端も音響検出とは別の値である。2秒を超えるtailは未確定のままとし、短縮しない。
EOFや欠測で窓が埋まらない場合は不完全観測とする。

releaseは実際に保持しているtoneのroutingとVoice世代に帰属する。存在しないtone、世代不一致、
既に終わったtoneへの指令を、確認済みreleaseとして扱わない。繰り返しreleaseによって進行中の
観測が置き換わる場合は旧記録をsupersededとして閉じ、未観測部分を無音で補わない。
発音時の観測とrelease観測は同時に存続できる。同じ256 pending／256出力の上限を共有する。
shutdownに伴う強制releaseも通常指令と区別して記録する。

検査は3身体×4 routingの独立renderとの照合、指令／活動／renderer終端の分離、
再release・退役・世代・容量・欠測・EOFと、off／observeの音声一致を含める。
この単位だけをI10全体の完成とはしない。

### I10全体の完了監査表

2026-09-20の範囲確定後の監査表は末尾の「I10の範囲確定と残作業の再定義」節にある。以下は2026-09-17時点の記録。

| 要件 | 必要な証拠 | 2026-09-17現在 |
|---|---|---|
| onset／release指令と私有実音 | 通常renderer→report／UI、独立実音照合、両bus、欠測・世代・容量、音声不変性 | release接続を追加。最終検査結果は後記 |
| 任意身体の六記述値と身体model世代 | 共通抽出式、2秒の因果窓、90%支持、最短0.1秒の公開間隔、固定尺度、若い個体・変更・routing・silenceの検査 | 通常runtime／report／UIへ接続。共有の窓集計とfeatures::Streamを利用し、独立PCM／スペクトル積分・身体世代・欠測・音声不変性を検査。開発尺度の適合はR4で別判定 |
| 動作条件付き自声予測 | モデル・尺度・時間窓・損失・学習率を実装前登録。予測凍結→実観測照合→更新、固定／動作非依存対照 | 活動／終端の5座標と将来の六記述値を発行時に固定して照合・更新する通常経路を接続。対照、対応変更、欠測・世代・容量・両bus隔離、通常binaryの実音と非干渉を検査。予測精度の研究上の採用は別判定 |
| 私有参加trace／credit | 固定方策の実行帰結と発行時のreference／anchor、非実行候補の非学習、支持・順序・退役・両bus隔離 | 有界Rust filterを移植し、Pythonの60積分条件・112更新・22 lookupと照合。最大16 traceの同時更新・退避、elapsed-only対照、帰還時の多峰保持も検査。共有inventoryと実Voiceの発行時凍結、元観測順の実帰結更新、full／elapsed-only予測の照合を通常renderer／report／UIへ接続した。検査結果は下記。候補生成への作用はまだ無効 |
| 候補帰結の数値経路 | 実身体descriptor→prototype／actual-Voice射影→既存の帰結列→診断。default差分、未知、支持範囲、反実仮想の非学習 | 距離・候補混合・ordinal backoff・時刻lookup・default差分の数値核を独立参照で検証済み。実音prototypeの適合と私有body／共有groupの通常descriptor照合も接続。候補featureと帰結表の通常接続は未完。未校正の列を生成へ有効化しない |
| I10の資源 | capture／extractor／予測／trace／候補の全費用、4／16／64 Voice、上限・欠落・鮮度を測定しR2へ渡す | 未完了。R2の全処理preflightやA3実機受入とは別 |

六記述値の正本は `docs/design-notes/dcc-neurocognitive-hierarchy.md` §9.6。
順序はspectral log2 centroid、spread、log2 RMS、positive envelope rise、positive spectral flux、
weighted accent densityであり、周波数指令や三帯域profileで置き換えない。
私有traceの既存数値参照は `scripts/temporal_cognition_reference.py::PrivateTimingTrace`。
自己音予測器とは別の状態であり、その保持則を自声予測学習へ流用しない。

今回のrelease実装では、観測slotに指令通番を加えて再利用時の二重加算を防ぐ。
同じVoice・toneでも古い指令handleは新しい窓へsampleを追加できない。
通常reportの各帰結はactionとscheduled_action_sampleを持ち、UIにもaction、最後の活動、
rendererが終端を初めて確認したsampleを表示する。欠測をまたぐrenderer終端は、
未観測区間内の正確な消音時刻を証明しない。

### release接続の検査結果（2026-09-17）

通常renderer、report、UIへrelease／shutdownの診断を接続した。3身体×4 routingについて、
対象tone単独の独立renderと最初・最後の活動sample、RMS、renderer終端が一致した。
他Voiceを混ぜた演奏でも観測off／onの両bus音声はbit単位で一致した。
再releaseによるsuperseded、世代不一致、tone欠落、EOFの未完窓、2秒窓で終端不明のtail、
欠測、slotの指令通番・Voice世代照合を検査した。shutdownが未開始toneを取り消す場合はcancelledと記録する。

通常binaryのoff／observe・report有無の音声一致、release記録のroutingと支持上限も通過した。
`cargo test`は1,017成功、0失敗、30ignore。
`cargo test exit=0 @ 2026-09-17T21:25:51+09:00`。全出力と終了コードを
`test_report.txt`／`test_status.txt`へ保存した。Clippy、全target check、format、diff検査も通過した。
検査対象の6ファイル（action_observation、schedule_renderer、report、runtime/mod、ui/windows、
tests/render_binary）のパス・NUL・内容・NULを上記順に連結したSHA-256: `3e233b6a70b452ed75ce69784815af9c9f879610fa6de0faf210a251161b769f`。

**release接続時点の再開記録。I10全体は未完了。** 続く実装単位として、私有のVoice別・bus別実音を有界captureから別workerへ渡し、
共有音群と同じ六記述値を身体世代・原支持付きで診断へ接続すること。新生・recipe変更・routing・
無音・欠測・若い窓を検査し、未校正の尺度を適合済みとして扱わない。
自声予測、私有trace、候補帰結、全I10資源検査は上の監査表どおり残る。

### 六記述値の接続契約（実装前登録、2026-09-17）

共有音群の既存ending窓集計を共通関数へ移し、私有音には既存features::Streamの全音源配分を適用する。
Voice別・bus別のcaptureを64個体、4個の再利用hop bufferに制限し、別workerでNSGTと特徴を更新する。
2秒の波形履歴を重複保持せず、NSGT ringと2秒の特徴・accent履歴を保持する。同じ窓集計の結果を検査する。
実時間経路は空bufferがなければそのhopを欠測として数える。offline rendererではbuffer返却を待てる。
reportの有無でcapture・解析・保持を切り替えない。offでは新経路を構築しない。

`temporal_body`に6成分の固定means／deviationsと既存accentの2成分means／deviationsを明示する。
共有音群も有効な場合はaccent尺度の一致を必須とする。未指定を適合済みの既定値で補わず、
未校正の診断設定として扱う。prototypeの適合・標準化と採用は後続の帰結表の工程で検査する。

同じVoiceのkind・unison・基礎ratio構造が変われば身体世代を更新する。新生／継承は新しいVoice世代として扱う。
既存の連続timbre／amplitude／pitch制御では世代を変えない。古いrecipeのtailと新recipeが重なるhopは
新身体の純粋な測定とは扱わず欠測にする。routingはtoneごとの実経路で集計し、未routingの寄与は既知ゼロとする。
2秒窓の90%支持とzero spectral massによるmaskを維持し、最短0.1秒間隔で原支持付きdescriptorを公開する。
出生／身体変更時の窓はそこから切り詰める。NSGTに必要な初期支持は別に検査し、架空の初期音を加えない。

### 六記述値を接続した動作

`temporal_cognition::body::Capture`を通常の`wire_runtime`から構築し、rendererでVoice別の
実sampleをrouting後に合算する。既存の私有音減算とは別の二bus観測で、共有P／ListenerTwinへ
IDや分離波形を送らない。64 Voice×二bus、4 hop bufferを再利用し、frameが欠けた場合は
次の実取得時刻から欠測を認識する。音声callbackに解析・確保・待機は加えていない。

別workerの各bus laneがNSGTと既存`features::Stream`を使い、
`observables::summarize`へ2秒の特徴・accent履歴を渡す。この集計は既存phraseのendingとも共有する。
新規の`temporal_body`設定が明示されたobserve演奏で有効になり、offではcaptureもworkerも作らない。
配置した固定means／deviationsをreportへ記録し、標準化済み値も同じ設定から出力する。
この設定は診断用であり、開発音源による適合・prototype採用はまだ行っていない。

kind・unison・基礎ratiosの変更は身体世代を更新する。既存の連続timbre変更は世代を維持する。
旧世代toneが新世代と重なる場合は、そのtoneが実際に通ったbusだけを欠測にする。
RecordはVoice世代、身体世代、原窓、依存支持の始点、利用可能時刻、bus、成分maskとcoverageを持つ。
workerが処理した時点のcapture時計を利用可能時刻へ反映し、queue遅延を原支持の更新と取り違えない。
`body_descriptor`／`body_observation` reportとUIの「Private body descriptors (uncalibrated)」へ接続した。

Recordの実サイズは128 byteで検査し、128件の固定配列は16,384 byteとなる。
48 kHz／512 hopで4個のcapture PCM bufferは1,048,576 byte。NSGT、特徴履歴、accent、
queue、snapshot等はこのPCM数値に含めていない。全費用と4／16／64 Voiceの並行負荷は残る資源検査で測定する。

検査内容:

- 実PCMから直接求めたRMS・log-envelope rise、独立したNSGT積分から求めたcentroid／spread／fluxと一致。
- 新生窓、完全無音、取得gap、身体世代変更、旧recipe混在、連続timbre変更、64個体上限、buffer枯渇を検査。
- sine／harmonic／modal×4 routingの独立renderで、対象個体のRMSとVoice世代を照合。
  別Voiceの混合音が存在しても、観測有無の両bus音声はbit単位で一致。
- 通常binaryのoff／observe・report有無でも音声一致し、presentation-only個体のhabitatは既知無音。
- 既定NSGTの長い窓で生まれたばかりの個体は、スペクトル支持の90%条件に達するまで該当成分が未知。
  2秒の一律warmupは設けていない。短い窓での動作検査にはnfft=2048／hop=512を明示した。

次の実装単位は動作条件付き自声予測。実装前にモデル、予測窓、特徴尺度、損失・更新則、
固定モデル／動作非依存モデルの対照を登録し、発行済み予測の採点後にだけ実行帰結で更新する。
六記述値の接続だけでは、自声予測・私有trace・候補帰結・I10全体の資源検査は完了しない。

六記述値接続の最終検査: Rust 1,023成功、0失敗、30ignore。
`cargo test exit=0 @ 2026-09-17T21:46:36+09:00`。Clippy、全target check、format、diff検査も通過。
検査対象は以下の順でパス・NUL・内容・NULを連結したSHA-256 `786019b22f0b97c18c338c62b8b23e1b261f58dc334094d24fead8824e58cde6`:

- `src/temporal_cognition/body.rs`
- `src/temporal_cognition/body/tests.rs`
- `src/temporal_cognition/observables.rs`
- `src/temporal_cognition/phrase.rs`
- `src/temporal_cognition/section/commitment.rs`
- `src/config.rs`
- `src/life/schedule_renderer.rs`
- `src/life/report.rs`
- `src/runtime/mod.rs`
- `src/viewdata.rs`
- `src/ui/windows.rs`
- `tests/render_binary.rs`

### 自声予測の第一数値モデル（実装前登録、2026-09-17）

動作条件付き予測の最初の接続では、既存の20 ms onset／2 s release窓について、
活動発生、最初の活動の相対遅延、活動がある場合のlog2 RMS、窓内renderer終端、
終端が確認された場合の相対時刻を予測する。これは実装上の観測量であり、知覚的終端ではない。
六記述値は発行前の身体入力として使う。将来の六記述値自体の予測・照合はこの第一接続の後に残し、
本接続だけで自声予測全体またはI10全体を完了としない。

数値モデルはVoice世代・身体世代・busごとの有界な線形残差モデル。
固定予測は現在の身体ampからamp/sqrt(2)をRMSの近似とし、最初の活動を1 sample後、
終端を合成器の既知envelope終端から予測する。予定終端は予測だけに使い、実音測定へ代入しない。
発生／終端の固定確率は0.99または0.01。時間値は窓長で割り、log2 RMSは20で割る。
窓内終端が確認できなければ終端時刻の教師値を作らない。

動作条件付きモデルは固定予測への残差を学ぶ。入力は切片、onset／release指標、
log2周波数/16、log2 amp/20、既知終端の相対時刻、発行前の六記述値と各有効mask。
六記述値は登録した固定尺度で標準化した値のtanh(value/4)を使う。
動作非依存モデルは切片と同じ過去の六記述値・maskだけを使い、動作種・指令周波数・amp・
予定終端を使わない。固定初期予測は発生0.5、遅延0.5、log2 RMS/20=-0.5、終端0.5、終端時刻0.5。

更新はmasked normalized LMS: w ← clamp(w + 0.1*(y - issued_prediction)*x/(1+||x||²), -8, 8)。
確率と相対時間の予測は[0,1]へclipし、RMS座標は[-8,8]へclipする。
発行済みの予測で平方誤差を算出し、同じ誤差で更新する。新しい重みで古い予測を採点し直さない。
固定対照は更新しない。モデル所有は最大64 Voice、pendingは既存256指令窓を使う。
身体世代が替わった後に届いた旧世代の帰結は新モデルを更新しない。
拒否・取消・未実行候補・欠測・未routing・未完窓は学習しない。完全観測の既知無音は発生の負例。

発行前の身体descriptorは同一世代、support end／availableとも発行時刻以前、
原支持から0.5秒以内の成分だけを使う。未支持座標は値0とmask0で表し、学習済み値で補わない。
この私有状態は共有P／ListenerTwinへ渡さず、生成作用も有効にしない。
固定／動作非依存対照、対応変更、既知無音と欠測、順序、世代、予測凍結、候補の非学習を検査する。

モデル容量が埋まった場合は最終発行command IDが最も古いモデルを退避する。同じVoiceの
身体世代変更ではそのスロットを置き換える。退避回数を記録し、旧所有者の遅着帰結は採点だけに使う。

### 将来の六記述値予測（追加実装前登録、2026-09-17）

六記述値の対象は、指令の予定時刻と発行時刻の大きい方から100 ms以上後にある最初の通常公開窓。
身体世代開始時刻を起点に、100 msをhop境界へ切り上げた公開周期から対象のendを発行時に固定する。
対象窓は既存の2秒・寿命／世代clipping規則に従い、指令後だけの短窓へ置き換えない。
観測値が失われた場合、より後の窓で代用しない。gapで公開格子がずれた場合も対象不一致として報告する。

各出力は同じ固定尺度による標準化値のtanh(value/4)。固定対照は発行前の支持済み身体記述値の
持続予測、未支持座標は0を冷始動基準とし、予測を観測値として扱わない。動作条件付き残差と
動作非依存対照には第一数値モデルと同じ18入力、masked normalized LMS、学習率0.1を使う。
動作非依存対照の初期出力は0、予測出力は全座標[-1,1]。観測maskのある座標だけで平方誤差と
更新を計算する。発行時予測、更新数、原支持、対象end、実際のavailable、未知理由を別reportへ出す。

六記述値用の重みは短い活動／終端モデルと分離するが、Voice／身体世代／busの所有と退避は共有する。
pendingとreport待ちはそれぞれ最大256指令、超過は計数して学習しない。先行予測の対象endより前に
同じVoiceの次指令が入った場合、その未照合窓をintervening_commandとして無効化し、固定した単一動作の
教師値へ混ぜない。取消、世代変更、対象窓喪失、EOFも未知として閉じる。非実行候補は発行しない。
通常経路でreport有無にかかわらず照合・更新・drainする。六記述値予測も生成選択には作用しない。


### 自声予測接続の検査結果（2026-09-17）

通常のrendererで受理したonset／releaseだけが予測を発行する。短い活動／終端予測は
`self_sound_outcome.prediction`、将来の六記述値は`self_sound_descriptor_prediction`に記録する。
後者は入力の支持区間・利用可能時刻・mask、発行時更新数、固定対象end、実観測Recordと実際の
available、三対照の保存済み予測・平方誤差、学習有無・未知理由を保持する。
UIには双方の発行・照合・更新・未知件数と六記述値のpending数を追加した。

- 人工的な動作―帰結対応を反転し、活動発生と将来記述値の両モデルで再学習を確認した。
  各条件1,600試行、末尾200試行の第1座標平均平方誤差は条件付きモデルが0.03未満、
  固定対照が活動で0.4超／記述値で0.2超、動作非依存対照が0.2超。
  これは数値回復検査であり、人の聴取応答への適合や実身体全般の予測精度ではない。
- 発行から照合の間に別の更新が入っても、発行時の値で採点する。既知無音は活動の負例として
  学習する一方、欠測・取消・非実行・未routing・旧世代は新モデルを更新しない。
  六記述値は未知座標をmaskし、対象窓が失われても後の窓へ付け替えない。
- Observerの古いslot／再処理、モデル64件の退避、六記述値pending／出力各256件の上限、
  身体記述値のavailable・世代・0.5秒の鮮度、bus間の学習隔離を検査した。
- 通常binaryではpresentationのみの固定sine身体を使い、実onsetのRMSと短窓予測の教師値、
  実release尾部の六記述値と固定した未来窓の照合を確認した。off／observe×report有無の
  4条件でWAV bytesは一致した。六記述値すべての支持を検査する身体はrelease=0.3秒とした。
  短い尾部が対象窓より前に消える場合はtarget_unavailableとなり、値を補わない。

最終検査: Rust 1,033成功、0失敗、30ignore。
`cargo test exit=0 @ 2026-09-17T22:09:21+09:00`。Clippy、全target check、format、diff検査も通過。
以下の順にパス・NUL・内容・NULを連結した検証版SHA-256は
`65f5744765a9b5d9323cae54f8d97a44afe6974cae821fc69028e98a549dea4b`。

- `src/life/mod.rs`
- `src/life/action_observation.rs`
- `src/life/self_prediction.rs`
- `src/life/self_prediction/descriptor.rs`
- `src/life/self_prediction/tests.rs`
- `src/life/sound/tone.rs`
- `src/life/schedule_renderer.rs`
- `src/life/report.rs`
- `src/runtime/mod.rs`
- `src/temporal_cognition/body.rs`
- `src/temporal_cognition/body/tests.rs`
- `src/ui/windows.rs`
- `tests/render_binary.rs`

### 次の実装単位

私有参加trace／creditを通常経路へ接続する。Python参照からの有界Rust filter移植は下記で完了した。
対象は§9.2の発行時reference／anchor保持、実onset／release帰結の一回限りの信用割当、
32 binの二head・16 trace上限、時間減衰と競合referenceの干渉、正規化lookupと診断。
自声予測の残差学習や既存OwnSoundHistoryをこのtraceの代用品にしない。

reference入力の現状: `Recall::matches_for`はgroup別の既存matcher結果を内部公開するが、
通常snapshotの`group_queries`は各queryのbest等に限られ、§9.2の最大16 referenceと
各7 anchor alternativesからなる凍結inventoryはまだない。共有workerで既存match・
acoustic support・period/anchorをまとめてinventoryを発行し、hop側で再DTWしない。
実装前に時刻支持・退役判定・封印順序・明示的な未適合パラメータを固定する。
候補帰結とI10全体の4／16／64 Voice資源検査も引き続き未完了。

### 私有trace接続の実装前登録（2026-09-17）

§9.2と既存Python `PrivateTimingTrace` を数値契約とする。最初にRustの有界filterと
独立Python oracleの照合を完成させ、続いて共有reference inventoryと通常rendererへ接続する。
数値移植だけを通常接続の完了とはしない。private traceのtau／kappa／strength上限は
既存episode保持パラメータと区別し、未適合の明示値を使う診断と研究採用を分ける。

Keyはbus epoch・episode ID／世代・periodic/nonperiodic family。referenceは最大16件、
各7 anchor代替、32 binのonset／release二head、nonperiodicはoverflowを1 bin追加する。
実行帰結の既知時刻区間と凍結anchorのuniform差分を積分し、欠測を他referenceへ再配分しない。
更新順は原観測区間end、onset→release、指令ID。全referenceを同時更新してから最小保持質量を
退避する。通常経路の封印・遅延整列はこの数値境界の外側で行い、逆順や重複を受理しない。
elapsed-only対照は同じ割当・減衰・bin上限を使い、競合reference干渉だけを外す。
空traceは生成lookupで未知。均等事前分布を使う場合はassay側で明示し、binに擬似観測を足さない。

現在の`Recall`は完全なretrieval score／availabilityを通常snapshotへ出していない。
inventoryをbest-matchだけ、負のDTW costだけ、現行16件への再正規化で代用しない。
共有memoryの保持・no-memory bias・正規化matchと、原観測のgroup assignment coverageを
接続した上でzを計算する。これはI9の全曲文脈を再開する依存ではなく、I10入力に必要な
既存memory契約の接続として扱う。


### 私有traceの有界Rust filter（通常接続前の数値検証）

`src/temporal_cognition/private_trace.rs`に、最大16 referenceと各7 anchor代替、
32 binのonset／release二head、非周期overflow、原時刻区間の指数保持積分、
信用割当・他referenceからの干渉・時間減衰・bin上限・安定退避・補間lookupを移植した。
最大32 trace分（既存16＋新規16）のscratchを初期確保し、全referenceの更新が終わってから
保持質量の最小順で16件へ退避する。更新中にこのbufferが拡張されないことを検査した。
referenceレコードは512 byte上限以内。runtime全体の資源合格を意味しない。

既存Python `PrivateTimingTrace` を変更せず、独立oracleから60積分条件、112状態遷移、
22 lookupをbit保持JSONへ保存した。各更新後の両headの全log質量、正規化値、信用、
未割当、退役・退避を照合した。時刻点／区間、負の相対時刻、多周期、線形overflow、
欠測anchor、部分支持、微小区間、長い絶対時刻、非常に長短のtau、空headを含む。
追加検査は同一順序キー／逆順の拒否、誤入力時の非変更、elapsed-only対照の閉形式、
16旧＋16新referenceの同時更新と安定退避、epoch失効、別の音群へ帰還したepisodeの多峰保持。
発行時予測の凍結や遅延deliveryの整列はこの数値filterの外側に残る。

生成器は`scripts/generate_temporal_private_trace_fixtures.py`。
fixture再生成のbyte一致と、保存したPython source SHA-256の現行source一致も確認した。
fixture SHA-256: `edee0b891442ee7c52bc17174734e7bbb51c0c4cd37f559fd7be5f000e9fdb34`。
比較中にはfixture生成器の可変list参照をcopyへ修正し、各更新時点のlog質量とretained keyを
その時点の値として保存した。Python oracleの式・更新規則は変更していない。

**通常runtimeにはまだ接続していない。** 現在は`#[cfg(test)]`下のnative filterであり、
共有inventory・実Voiceの発行時保持・実音の信用更新・report／UIに到達した証拠とは区別する。
`Recall::matches_for`から既存group別matchを読む経路はあるが、`recall::retention`も現状は
test buildのみ。次はsealed commitmentとcached coarseの受信にこの保持状態を接続し、
原支持・availabilityとnormalized matchを使うretrieval scoreを通常workerから公開する。
その値とgroupごとの2秒assignment coverageを使って、欠測massを再配分しないinventoryを作る。

追加の`cargo clippy --tests -- -D warnings`は既存の15警告で失敗した。
内訳は`src/life/temporal_participation.rs`の12個の整数cast、`src/runtime/mod.rs`の
剰余比較1個、`src/temporal_cognition/recall/retention.rs`の条件式2個。
新規private_traceからのClippy診断はなく、通常の必須`cargo clippy -- -D warnings`は通過した。


この数値移植版の最終検査: Rust 1,040成功、0失敗、30ignore。
`cargo test exit=0 @ 2026-09-17T22:27:10+09:00`。通常Clippy、全target check、format、diff検査は通過。
次の順でパス・NUL・内容・NULを連結したSHA-256は
`323fca7273def3a7f0b422826ec3190b7b8099fe4ed420d3652564a206c9e2ba`。

- `src/temporal_cognition.rs`
- `src/temporal_cognition/private_trace.rs`
- `src/temporal_cognition/private_trace/tests.rs`
- `scripts/generate_temporal_private_trace_fixtures.py`
- `tests/fixtures/temporal_cognition/private_trace.json`
- `scripts/temporal_cognition_reference.py`（照合に使用、変更なし）

### 共有memory保持状態の通常接続（実装前登録）

`temporal_memory.retention`を明示した場合だけ、既存の取得clock・sealed commitment・cached
coarse結果をnative `Retention`へ渡す。tau、kappa、strength上限、r_max、no-memory biasは
明示値とし、暗黙の適合済み既定値を作らない。音響match項は既存`temporal_section`の
14残差特徴、固定平均／偏差、15係数をそのまま共用する。別の負DTW costモデルを導入しない。

現在の通常bankは各sealed occurrenceを新しいepisodeとして保存する。その正の割当supportを
初期strengthとし、検索結果だけで過去episodeのrecurrenceを捏造しない。保存済みepisodeへの
明示的なrecurrence割当経路は別途接続が必要であり、出生だけの更新を完全な保持学習と呼ばない。
既存episodeに対する干渉は、そのcommitmentに封印済みのcoarse距離だけから計算する。
coarse欠落は上下限を広げ、現在の検索結果で過去のcoarseを補完しない。
容量退避は保持量上限が最小のepisode、同値は初回commit時刻とhandleで決め、
descriptor・transport・graphの所有を同時に退役させる。

通常診断は取得clock、初期strength、原観測・封印・受信時刻、rate envelope、
retrieval score／確率の上下限を公開する。各groupでは同じepisodeの最大音響scoreを使い、
no-memory biasを含めて正規化してから最大16件を公開する。切捨てmassを再配分せず記録する。
この接続だけではprivate reference inventory、私有trace更新、候補帰結、研究上の採用は完了しない。


### 共有memory保持状態とretrieval scoreの接続結果

`recall::retention`と取得clockのcontextを通常buildへ接続した。明示した
`temporal_memory.retention`に対し、取得clock、原時刻順のsealed commitment、
そのcommitment用に凍結したcoarse結果から、支持量付きepisode出生・干渉・経過減衰・
取得欠測の上下限を更新する。容量到達時は保持量上限が最小のepisodeを退避し、
同じidentityのdescriptor、transport、graph、cached matchを失効させる。

`recall::retrieval`はSectionが既に使う14残差特徴の算出を共用する。正規化した音響scoreと
保持量、同じ明示no-memory biasから、episode別確率とstored-match総確率の上下限を計算する。
各groupで最大16件を公開する前に全取得候補で正規化し、切捨て質量を別記する。
16件に再正規化しない。上下限は不確実性の範囲であり、上限の合計を単一の確率分布と
解釈しない。現在の値は受信済みのlive contextであり、過去の照合時点へ遡及しない。

reportの`temporal_observation.memory`に`retention`と`retrieval`を追加し、UIには保持件数・
出生数・退避数・rate状態・検索確率上下限を表示する。`owned_bytes`はRetention内部の
配列とscratchだけを含む。親Recallのscratch、clock、descriptor、snapshotコピーや他の
I10経路を含む総資源量ではない。全体測定は残る資源検査で集計する。

単体検査では、取得済み支持からの出生strength、時刻因果、検索だけではstrengthを
増やさない点、16件を超える結果の切捨て質量、不正な明示パラメータを検査した。
別の欠測条件では1/8支持のepisodeが、より古い完全支持episodeに先立って退避され、
cached matchからも消えることを確認した。既存Python fixtureによる保持filter検査は維持した。

通常rendererでは既存の観測設定を使い、4.6秒の固定seed・presentation-only入力を
追加した。whole診断は外し、I9の再開を依存にしない。複数のsealed episodeと既知coarse距離、
保持量・検索確率がreportへ届き、無音habitatにはepisodeが生まれないことを確認した。
off／observe × report有無の4条件でWAVはbyte一致した。

この時点で通常bankが行う正の保持更新は、各sealed occurrenceを新しいepisodeとして
保存する出生割当である。既存episodeへの明示的な再帰割当は未接続であり、検索結果を
その代わりに使っていない。private inventory・実Voiceへの発行時凍結・実行帰結のtrace更新、
候補帰結と全I10資源検査も未完了。private_trace自体は引き続きtest buildのみ。

次の接続で読む場所は、`recall::Snapshot::retrieval`、
`proposals::frontend::Snapshot`のgroup handles／eligible／features／energy、
`proposals::frontend::recurrence::GroupSnapshot`のperiod情報。現在のaccent summaryは
最後のanchor時刻を公開しておらず、2秒の物理取得・association支持の履歴もinventory用には
未集計。§9.2のcoverageとanchorをこの元観測から作り、現在のhop値だけで代用しない。

最終検査: Rust 1,042成功、0失敗、30ignore。
`cargo test exit=0 @ 2026-09-17T22:48:35+09:00`。全出力と同一shell終了コードを
`test_report.txt`／`test_status.txt`へ保存した。通常Clippy、全target check、format、diff検査は通過。
次の順でパス・NUL・内容・NULを連結したSHA-256は
`45f5b2d67a5e8d7728c75a71925c81f8da3821d02d5125f6847d3f42ec6b3f67`。

- `src/config.rs`
- `src/temporal_cognition/recall.rs`
- `src/temporal_cognition/recall/clock.rs`
- `src/temporal_cognition/recall/retention.rs`
- `src/temporal_cognition/recall/retrieval.rs`
- `src/temporal_cognition/recall/tests.rs`
- `src/temporal_cognition/section/runtime.rs`
- `src/temporal_cognition/observation.rs`
- `src/ui/windows.rs`
- `tests/render_binary.rs`

### 共有reference inventoryの通常接続（実装前登録）

§9.2のinventoryを、既存のgroup assignment・accent・period・retrievalだけから構築する。
2秒窓はepoch開始で切り詰め、物理取得と有効associationが揃う区間のsample数をcの分子、
窓全体を分母とする。既知の非活動も支持に数える。aは既存のtrajectory割当をhop内で
全trajectory数により平均し、元の支持時間で積分した後、同じ窓内のresolved groupで正規化する。
振幅による重みは加えない。窓内で退役したgenerationの割当も分母に残し、現行groupへ
再配分しない。generation再利用は旧履歴を継承しない。

retrievalの確率が上下限になる場合、inventoryも両端を記録する。確定して割当可能な信用は
下限に限定し、その選択を`credit_policy = lower_bound`として公開する。上限との差を確定質量と
見なさない。下限を取り直して正規化せず、no-memory biasは同じ保持設定への参照を明記する。

anchorはそのgroupに実際に観測された直近のsalient accentの元hop区間。欠測・association欠落を
またいだanchorは使わず、次の観測accentで初めて更新する。周期は既存arrivalと同じ最大支持の
period peak（同値は短い周期）を、原支持と利用可能時刻を検査して再利用する。周期を持たない
anchorはnonperiodicとし、Voice固有周期は共有workerで作らず指令発行時に凍結する。
同じepisode／世代／familyへの複数group bindingは別anchorとして保持し、平均時刻に潰さない。
最大112 bindingから16 referenceを公開し、検索側とinventory側の切捨てを別記する。

各referenceは7 anchor代替を含め512 byte以下とし、2秒の支持ring・scratch・公開snapshotの費用を
別途数える。report／UIへ通し、空・欠測・epoch・退役・帰還・複数anchor・切捨て・独立した式照合を
検査する。この接続では実Voiceへの発行時凍結・trace更新はまだ完了としない。


### 共有reference inventoryを接続した結果

`reference_inventory::Stream`を共有observerの各busに接続した。retention設定の有効時だけ
2秒支持ringと最大112 bindingのscratchを構築し、epoch変更時には空へ戻す。
新しいworker計算は既存のacoustic／period／memory結果を読み、追加のDTWやVoice別解析をしない。
reportの`temporal_observation.reference_inventory`と、同じsnapshotを読むUIへ到達した。

各hopの原支持から観測sample数とfractional assignmentを積分し、窓の分子・分母・
epoch clipping・group世代を公開する。退役がそのhop末に決まっても、当該hopの旧group割当は
分母に残す。新しいgroupは旧slotの支持・anchorを引き継がない。支持が完全な非活動hopは
coverageに入り、割当なし・欠測hopは正のassignmentを作らない。

各referenceはepisode ID／世代／familyをkeyとし、複数groupのanchorを最大7個保持する。
周期が有効なgroupは既存period peakを使い、非周期にはVoice固有周期の未確定を`None`で残す。
時刻支持は元のaccent区間のまま。取り除いたanchorの質量は未割当へ残す。保持下限からの
信用と上限を両方公開し、`credit_policy`と同じno-memory biasの設定参照を明記する。
検索段階とinventory段階の切捨ては別に記録し、最後の16件を再正規化しない。

5単体検査で、独立したa×c×確率の算式、2種類のanchorへの質量配分、2秒／epoch clipping、
欠測と既知の非活動、退役直前hop、slot再利用、同じepisodeの新group binding、周期の異なる
二anchor、未来の周期証拠の不使用、16件切捨て、bias不一致とepoch不一致を照合した。
支持ring・scratchのcapacityが長い更新で増えない点も確認した。

通常rendererの4.6秒検査でも、実音→sealed episode→retrieval→reference／anchorを
reportから確認し、元anchor時刻が利用可能時刻より後にならない点と、anchor重み和・
割当／未割当保存を検査した。off／observe × report有無の音声byte一致を維持した。
各referenceの512 byte上限と16件の8,192 byte上限は検査済み。
`owned_bytes`は支持ringとscratchを含むStream、`reference_bytes`は公開reference配列の
費用を表す。Snapshot全体と転送コピーの費用は別であり、I10全資源の合格ではない。

今回の通常接続は共有inventoryまで。私有traceの数値Bankはまだtest buildのみ。
次は指令発行時にこのinventoryとVoice固有周期・予測を凍結し、実onset／release帰結を
元観測end順で一回だけ更新する。epoch／episode退役を、単に現在matchしなくなった状態と
混同しないretained-handle連携も必要。候補帰結・全I10資源検査、既存episodeへの明示的な
再帰割当の未完状態は維持する。I9は再開していない。

最終検査: Rust 1,047成功、0失敗、30ignore。
`cargo test exit=0 @ 2026-09-17T23:03:17+09:00`。全出力と終了コードを同一shellで記録した。
通常Clippy、全target check、format、diff検査は通過。次の順でパス・NUL・内容・NULを
連結したSHA-256は`4c0faeb443e0f066f7b5e6ce486a5d4a72ebe9b0434d556c895575abf284753e`。

- `src/temporal_cognition.rs`
- `src/temporal_cognition/reference_inventory.rs`
- `src/temporal_cognition/reference_inventory/tests.rs`
- `src/temporal_cognition/private_trace.rs`
- `src/temporal_cognition/recall.rs`
- `src/temporal_cognition/recall/retrieval.rs`
- `src/temporal_cognition/observation.rs`
- `src/ui/windows.rs`
- `tests/render_binary.rs`

### 実Voiceの私有trace接続（実装前登録）

`temporal_private_trace`のtau／kappa／strength上限を明示したobserve実行だけで有効にする。
共有retention／inventoryを必須入力とし、generatorの私有traceはhabitatの実帰結だけで更新する。
presentation-only実行は私有参加への信用を持たない。両busの既存自己音測定は維持する。

共有observerはinventoryと同時点のretained episode ID／世代を別のimmutable contextで公開する。
保持集合の配列はepisode出生・退避時だけ更新し、hop側で全bankを複製しない。
現在のqueryに現れないepisodeを退役と見なさない。利用可能時刻が発行より後のcontext、
0.1秒を超えて古いcontext、epoch不一致は信用入力にしない。

指令発行時に最大16 reference、anchor、Voice世代、固有周期、およびfull／elapsed-only両Bankの
head確率を凍結する。非周期にはCoupling clockの現在のintrinsic rateまたはParticipation clockの
base rateを使い、固有周期を持たないclockはそのreferenceを未知にする。共有thetaを私有周期の
代替にしない。periodic referenceは共有観測周期をそのまま使う。

実onsetはhabitat sampleが閾値を初めて超えたsample区間、実releaseは活動確認後にrendererで
確認した終端sample区間に帰属する。結果支持の既知割合を信用へ掛け、不完全な帰結は確定した
onset／endがある部分だけに限定する。未実行、拒否、取消、superseded、無音、未確認終端に
正の信用を足さない。観測器の指令IDによる一回限りの更新を保ち、未完指令の最早時刻を
watermarkとして、完了結果を元観測end・onset→release・ID順へ封印する。EOFは残る指令を
不完全として閉じてから既知結果を整列する。待機と容量超過を診断へ別記する。

64 Voice、256 pendingの固定容量を使い、最初のconfirmed creditだけで空traceを作る。
予測は発行時に存在したheadを使い、空headは生成用に未知のまま、診断損失だけで明示した
32／33 binの均等事前分布を使う。信用加重soft-bin損失は診断であり物理観測likelihoodとは
呼ばない。候補の非学習、順序、退役、世代、routing、固定方策、両対照、音声不変性を検査する。


### 実Voiceから私有traceまでの通常接続結果

`life::participation_trace::Bank`を既存の私有自己音観測器に接続し、
`temporal_cognition::private_trace`の数値filterを通常buildへ移した。
明示した`temporal_private_trace`設定のobserve実行で、64 Voice・256 pendingの固定容量を
構築する。各Voiceにfullとelapsed-onlyのBankを持ち、初回の正の実帰結信用で初めてtraceを作る。
発行・完了・信用・未割当・保留・容量／出力drop・退役・誤順序等の累計をUIへ公開する。

共有observerは別の`reference_context`で、同時点のinventoryとimmutableな保持episode集合を
公開する。保持集合の配列を作り直すのはepisode出生・退避時だけで、hop側はArcを共有する。
これは全bankをhopごとに再検索する経路ではない。現在のmatchがないepisodeも保持集合に
残る限り有効であり、実際の退避またはepoch変更だけで私有Bankから外す。
source側のepochを混ぜず、未来のcontextを適用しない。発行時または封印時に0.1秒を超えて
古いcontextは正の信用に使わず、最新の受信を待つために既知の結果を改変しない。

VoiceのPhonationBatchに、その時点のprivate clockの固有周期を載せた。
Couplingは現在のintrinsic rate、Participationはbase rateを使い、ThetaGate等は未定義のまま。
releaseでも元toneのroutingを保ちつつ、指令batchの現在の固有周期を渡す。
発行時にはreferenceとanchorを値として固定し、両Bankの確率を保存する。後のcue変更、
別の候補、後から生じた学習値で、この予測や割当を置き換えない。

自己音観測結果へ`contiguous_observed_end_sample`を加えた。窓の先頭から連続して取得できた
支持を追跡し、欠測をまたいで初めて見つけた活動やrenderer終端を、正確な発生sampleとは
扱わない。時刻がこの既知prefix内なら、そのsample区間を使い、窓全体の既知割合を信用へ掛ける。
onset確定後の欠測は既知の時刻を消さず、onsetを隠した欠測は正の信用を生まない。

完了した結果は、未完の指令が持つ最早の予定時刻をwatermarkとして保留する。
解放時には原観測end・onset→release・指令ID順に整列し、数値Bankへ一回だけ渡す。
EOFでは残る実観測を不完全として閉じてから封印する。世代交代は同じIDの旧Voiceモデルを
置き換え、旧pendingから新Voiceへ信用を渡さない。未実行・取消・superseded・無音・
presentation-only・未確認終端は私有参加を強化しない。

reportへ`private_participation_trace`を追加した。発行時刻、inventory時刻、固有周期、
元観測区間、封印時刻、各凍結reference／anchor、full／elapsed-onlyの発行時確率、
実結果のbin配分、信用、未割当、soft-bin診断損失を保存する。空headには診断用の
32／33 bin均等事前分布を明示し、観測済みheadのゼロ確率は`zero_probability`として
区別する。この損失は物理的観測likelihoodでも人の参加意欲の検証でもない。

7単体検査で、発行後のcue変更、検索だけの非学習、指令ID重複、同じVoiceの次回予測、
遅延releaseと後続onsetの原時刻整列、複数結果の発行時予測保持、episode／epoch／Voice世代退役、
routing、固有周期なし、未来／古いcontext、取消、欠測の前後差、256 pending上限と逆順delivery、
競合referenceによるfullとelapsed-onlyの予測差を確認した。

通常rendererの最初の単独entrained素材ではgroup入替えでanchorが支持されず、信用はゼロだった。
未知を埋めず、その結果を保持した。正の経路の検査には、既存検査で支持が得られた220 Hzの
反復素材を2秒先行させ、880 Hzの対象Voiceを加える固定seedの8.1秒素材を用いた。
実音から得たinventory→発行時凍結→実帰結信用→同じVoiceの次回の学習済み予測を、
通常binaryのreportで確認した。既存4.6秒のpresentation-only素材では信用・学習ともゼロ。
両素材それぞれoff／observe × report有無の4条件で音声はbyte一致し、完了時pendingとerrorsは0。

これは私有traceの通常往復と非干渉の技術検査であり、I11の生成作用を有効にするものではない。
明示値は未適合の診断設定である。候補帰結表、4／16／64 Voiceでの全I10資源測定はまだ未完了。
保持集合の更新配列、64×2個の数値Bank、凍結予測と待機結果、reportレコードと転送コピーも
資源測定の対象に含める。既存episodeへの再帰割当の制限とR/A未達は、前節の境界を維持する。

私有trace接続後の全Rust検査は1,054成功、0失敗、30ignore。
`cargo test exit=0 @ 2026-09-17T23:27:02+09:00` を記録した。

### 候補帰結の数値核（実装前登録）

次の単位では§9.6の候補射影を、通常接続前の独立した数値核として検証する。
六記述値の共通座標RMS距離・厳密な同距離tie・0.25境界、最大128代替の
候補条件付き重み、未支持代替の未知質量、解釈entropy、三ordinal headの
混合後temperature／固定issue coverage／prior backoffを対象とする。
prototype選定とgroup選定は同じ距離規則を使い、既存身体Recordの標準化と照合する。
同一身体のIDやpitch指令を欠けた記述値の代替にしない。

32 cellの絶対sample時刻lookupは最寄り一cell、厳密な中点は前cellとする。
未知cellを飛び越えず、horizon外を外挿しない。各signed列は候補と身体defaultの
両側支持がある場合だけ差分を計算する。ゼロと未知を分離する。
Pythonの独立数値参照と境界・欠測・極端なscoreの検査を行い、未実行候補を
passive state、私有trace、自声予測器へ入力しない。この核だけでは通常経路、
prototype適合、feature射影、反実仮想精度、資源測定の完了とはしない。

### 候補帰結の数値核の検査結果

`temporal_cognition::consequence`をtest buildへ追加した。通常buildにはまだ接続していない。
実身体Recordの固定尺度・maskによる標準化結果を入力し、最大8候補から、共通座標の
RMS距離→共通座標数→安定IDの順で選ぶ。共通座標なしは未知、0.25は適合境界に含める。
1 ULPだけ異なる距離をtieへ丸めず、範囲外や非有限入力を適合扱いにしない。

候補条件付き分布は最大128代替の元joint質量を保持する。未支持のaction射影は未知へ移し、
支持された代替だけをscore差で傾ける。解釈entropyとordinal categoryのentropyを混同しない。
三headは既存`ratings::project_rating`を使い、head別の欠測・候補混合・temperature・
issue時点のcoverage・固定priorを順に処理する。表示用backoffが計算できても、
headの支持がゼロならsigned列は未知のままとする。関係scoreは呼出側が別に与える入力であり、
現時点ではfeatureからscoreを作る経路を実装したとは扱わない。

Pythonの80桁Decimalによる確率空間の独立参照を60条件生成し、Rustの対数空間計算と照合した。
同率、集中、単一、全未知、head別欠測、ゼロ質量、極小質量、128代替、coverageゼロ／全支持を含む。
最初の照合で、集中した分布の微小entropyに丸め差を検出した。最大成分以外の質量を
`ln_1p`へ入れる形で修正し、全条件を通過した。巨大な共通scoreは先に差し引き、
元のjoint重みが加算時の丸めで失われないようにした。scoreの変更だけでは元の未知質量を
回復できないこと、繰り返した候補評価が入力を変えないことも確認した。

32 cellのlookupはu128中間値を使う絶対sample演算とした。u64上限近くの192,001時刻を、
32候補を全探索する独立の距離計算と照合した。中点tie、奇数幅、範囲外、選択cellが未知の場合を
含み、未知を飛び越えたり隣接cellと混合したりしない。signed差分は各列で両側支持を要求し、
片側欠測の両方向と、同一defaultの既知ゼロを区別した。

6単体検査が通過。fixture SHA-256は
`5f6a5f9edfe03a63f7a0ca97abdda3fbfcafb247d0c16c70ca485a508c36f4fc`。
生成は`python3 scripts/generate_temporal_consequence_fixtures.py`、核の検査は
`cargo test temporal_cognition::consequence --lib -- --nocapture`。

次に必要なのは、実音development recordによるPAM prototypeと射影範囲の保持、
共有groupの六記述値cache、featureごとのobserved／hypothetical／unsupportedのscratch射影、
7 action×32 cell×最大8 prototypeの表、実Voice側のdefault対応・世代／bus／鮮度検査、
report／UIへの通常接続である。私有traceと局所own-excluded energyの列も各自の支持を保って結ぶ。
今回の独立核や合成fixtureを、実音反実仮想精度・ordinal校正・I11生成作用の採用とはしない。
4／16／64 Voiceでの全I10資源測定も引き続き未完了。

最終検査は全Rust 1,060成功、0失敗、30ignore。
`cargo test exit=0 @ 2026-09-17T23:40:17+09:00`。全出力と終了コードを同一shellで保存した。
通常Clippy、全target check、format、diff検査が通過した。
次の順でパス・NUL・内容・NULを連結したSHA-256は
`05fdb83c1fed4e32acbfe90b408c66ca6b6c461a846d4c04afa7354f4fa11ecd`。

- `src/temporal_cognition.rs`
- `src/temporal_cognition/consequence.rs`
- `src/temporal_cognition/consequence/tests.rs`
- `src/temporal_cognition/ratings.rs`
- `scripts/generate_temporal_consequence_fixtures.py`
- `tests/fixtures/temporal_cognition/consequence.json`

### 実音prototypeモデルの接続契約（実装前登録）

候補表の前提として、実音descriptorから固定尺度とPAM medoidを生成するオフライン処理を追加する。
最初の診断corpusはsine／harmonic／modal、110／440 Hz、振幅0.04／0.12、
both／habitat-only／presentation-onlyの36素材、固定seed、48 kHz・2048 FFT・512 hopとする。
2秒sustain後にreleaseし0.6秒待つ。各busについて0.2／0.8／2.0／2.2秒の直前に公開された
最新descriptorを採用する。距離や結果を見て素材・cutを選び直さない。
切詰めた若い窓、sustain、release、既知無音を含み、欠けた座標を埋めない。

固定尺度は全登録recordの各座標で支持された値の平均と母標準偏差とし、分母は各座標の件数。
標準化時だけsdを1e-6で下限化する。距離は既存の共通座標RMS。
PAMはID辞書順の最小recordからfarthest-firstで初期化し、同率は同じID順。
最良の目的値減少swapを選び、改善1e-6以下または100 passで停止する。
同じ目的値のswapは得られるmedoid ID列の辞書順で決定する。
record間で共通座標がなく、選んだmedoidで覆えない集合はエラーとして残す。

モデルには原record・素材・reportのhash、尺度、medoid ID、停止理由、割当と距離、
診断用の0.25境界を保持する。モデル版はそのJSON内容のSHA-256で固定する。
通常runtimeでは明示した`temporal_body_prototypes`だけを読み、既存の`temporal_body`尺度、
accent尺度、sample rate／FFT／hopの一致を要求する。最大8 medoidを起動時に標準化する。
身体観測workerで新しい実descriptorの公開時だけ照合し、同じslot・身体世代・busのrecordと
対応させてreport／UIへ出す。未知は未知として残し、report有無で照合を切り替えない。
この照合だけでは候補のpitch／envelope／rate射影範囲や反実仮想精度の合格とはしない。
今回の診断corpus、未適合accent尺度、同じcorpusへの距離は研究上の検証・held-out評価ではない。

### 実音prototypeモデルの通常接続結果

`scripts/collect_temporal_body_prototypes.py`で登録済み36素材を実renderし、
両bus・4 cutの288 recordを取得した。欠落素材、capture drop、invalid hop、容量超過はなかった。
`scripts/fit_temporal_body_prototypes.py`は固定尺度とPAM medoidを作り、
原descriptor、recipe、原窓、各素材・report・WAVのhash、割当と停止理由をモデルへ保存する。
PAMの初期集合と出力medoidはID順で保存し、探索は登録したfarthest-firstと最良swapに従う。
8 medoidは7 passで改善1e-6以下の条件に到達した。目的値は31.01300950683909。
原モデル版は`b3eb8cba69007e16aedb013d9ee8ba617b3ecb4d3420afb492768bfeba02fd28`。

`body_model`の距離関数を通常buildへ移し、`temporal_body_prototypes`を明示した場合だけ
私有身体workerへ渡す。medoidは起動時に一度標準化し、新しい実音descriptorが公開される時に
最大8件を走査する。body record自体へ値を追加せず、同じSnapshotのparallel配列に照合結果を持つ。
recordのslot・身体世代・busと同時に更新／消去し、世代交代の初期未知窓や退役時に旧割当を残さない。
reportは`body_descriptor.prototype_assignment`、Snapshotは32 byteのモデル版を持つ。
assignmentのkeyはモデル内でID順に並べたmedoidのゼロ始まりindexと0の組であり、Voice IDではない。
UIも同じ計算済み結果とモデル版prefixを読む。reportの有無で処理を切り替えない。

runtime設定は最大8件、ID順・一意性、有限値・mask、尺度、sample rate／FFT／hopと
モデル版の書式を検査する。完全なモデル内容のSHA-256とcorpusの対応はオフライン検証器で照合する。
runtime内で未提供の全development artifactを検証したとは扱わない。

`scripts/verify_temporal_body_prototypes.py`で8 medoidモデルを通常rendererへ渡し、
36素材すべてを再実行した。WAVはprototype無効時と全件byte一致した。
重複を除く公開descriptor 1,512件のmedoid、共通座標数、距離、未知判定が独立計算と一致した。
適合1,272件、未知240件。鳴っているbusではsine 81/84、harmonic 38/84、modal 73/84が適合し、
routing三条件で同じ結果だった。非routing busの既知無音も記述値として照合するが、
それをlistener作用や発音成功の支持へ転用しない。harmonicで多い未知を除外せず記録した。

同じ288登録record上の感度検査は次のとおり。閾値変更や16 medoidの本番採用は行っていない。

| medoid数 | 0.125以内 | 0.25以内 | 0.5以内 |
|---|---:|---:|---:|
| 4 | 144 | 176 | 228 |
| 8 | 184 | 244 | 288 |
| 16 | 228 | 288 | 288 |

これは学習corpus内の距離分布であり、候補rankや未知身体への転移の精度ではない。
モデル・全選択record・感度検査・通常再実行結果を[I10 prototype記録](i10-body-prototypes.json)へ保存した。
原reportとWAVは`target/i10-body-prototypes-20260917/`に保持し、再生成コマンドも記録へ含めた。

Pythonの4検査は小型全組合せの目的値、入力順不変性、maskと尺度、非連結record、
モデル版・TOML出力、無効recordの非除外を確認した。Rust側では設定不一致と対応変更に加え、
実workerの世代交代・初期未知・退役時の割当消去を検査した。通常binaryの検査は、
実renderから採ったmedoidと同じ身体の対応、非routing busの非適合、モデル版、
off／observe×report有無の音声一致を確認した。JSON→TOMLで1 ULP程度の差が生じるため、
この往復の距離照合は1e-14以内とした。0.25境界の選択規則や同距離tieを緩めたわけではない。

候補帰結表はまだ通常接続されていない。次は共有group descriptor cacheと
observed／hypothetical／unsupportedのfeature射影、7 action×32 cellの表、
実Voiceのdefault対応と各列の支持・世代・鮮度を結ぶ。prototype照合を候補の射影範囲の
適合証明に置き換えず、私有trace・局所energy列と両busの帰結を接続する。
4／16／64 Voiceの全I10資源測定、反実仮想精度・ordinal校正・I11採用は未完了。

最終検査: 全Rust 1,064成功、0失敗、30ignore。
`cargo test exit=0 @ 2026-09-17T23:57:35+09:00`。全出力と終了コードを同一shellで保存した。
Pythonの4検査、通常Clippy、全target check、format、diff検査も通過した。
次の順でパス・NUL・内容・NULを連結したSHA-256は
`e04816e5fe1bf4efd2e7e6b675d248d24dbfedac4050d389c3f8d25a9969f2df`。

- `src/config.rs`
- `src/temporal_cognition.rs`
- `src/temporal_cognition/body_model.rs`
- `src/temporal_cognition/body.rs`
- `src/temporal_cognition/body/tests.rs`
- `src/temporal_cognition/consequence.rs`
- `src/life/self_prediction/tests.rs`
- `src/life/report.rs`
- `src/runtime/mod.rs`
- `src/ui/windows.rs`
- `tests/config_restore.rs`
- `tests/render_binary.rs`
- `scripts/collect_temporal_body_prototypes.py`
- `scripts/fit_temporal_body_prototypes.py`
- `scripts/verify_temporal_body_prototypes.py`
- `tests/test_temporal_body_prototypes.py`

### 共有音群とprototypeの対応（実装前登録、2026-09-18）

phraseが保持する音群別の実観測historyとaccentを再利用し、
J=[max(epoch開始、group世代開始、t-2秒),t]の六記述値を共通summarizeで集計する。
foregroundのphrase開始では窓を切らず、phraseの仮説や終了を実音支持にしない。
`temporal_body_prototypes`とphrase処理が両方あるobserve実行で、100 ms以上の間隔ごとに
最大7群のdescriptorを一度集計し、固定尺度で最大8 prototypeとの対応を求める。
この追加経路でスペクトルを再走査したり、履歴を複製したりしない。

候補は現在eligibleなresolved groupだけとし、residual、別bus／epoch、未来の支持、
退役groupを選ばない。距離0.25と既存の同距離tieを使う。公開間隔の途中でも
retained集合やeligibilityから失われたgroupのcacheと対応を消す。その際に時刻を更新したり、
別の便利なgroupへ対応を付け替えたりしない。epoch変更や上流失敗では全cacheを無効にする。
snapshotはbus／epoch、原支持、モデル版、group別descriptorとprototype別対応を持ち、
report／UIで確認する。私有Voice ID／分離音は入力に使わない。
候補feature・帰結表・実Voice消費者・全資源測定は引き続き後続の必須工程とする。

### 共有音群descriptorとprototype対応の通常接続結果

`Phrase::body_descriptors`が既存のhistory／accentから共通`observables::summarize`を呼び、
group世代から切り詰めた2秒窓を返す。新しい履歴ringやスペクトル解析は追加していない。
固定尺度の標準化も私有Record・medoid・共有groupで共通化した。
prototypeの内部保存は最大8件の固定配列へ変更し、worker起動時のheap配列を不要にした。

`body_model::Shared`はprototypeモデルとphraseが有効な場合だけ、両busの観測workerで
100 ms以上の間隔ごとに最大7群を集計し、prototype別に最寄りの適格groupを選ぶ。
private body inputと同じ6座標・距離0.25・座標数と安定handleのtie規則を使い、
選ばれるgroupはそのbus／epochのresolved pool内に限る。
私有音・Voice ID・作者labelを受け取る経路は追加していない。

毎hopの有効性確認は、対応groupがretained／eligibleでなくなった時に直ちにcacheから消す。
次の集計時刻までは別groupへ再割当せず、元のend_sampleも更新しない。
epoch変更、phrase／acoustic失敗、workerの異常終了でcacheを無効化する。
同じepochの正常EOFでは最後の実観測窓を診断として残すが、EOF時刻へ支持を延ばさない。

observer snapshot versionを23に更新し、`group_prototypes`にモデル版・bus／epoch・
実窓descriptor・最大8個の対応を公開した。対応配列のindexがmedoid index、
各assignment.keyは`[epoch, group_generation]`であり、私有body側のindex形式とは別である。
reportは既存のtemporal observation経路を使い、UIは同じsnapshotの対応数・descriptor数・
支持endを表示する。候補帰結や生成作用はまだ接続していない。

新しい単体検査では、若い窓、2秒切詰め、energy重み付きcentroid／spread、
時間平均energy／rise／flux、欠測での90% maskを独立計算と照合した。
foreground仮説を変えても音群descriptorが変わらないことを確認した。
別の検査で100 ms未満の再集計抑制、支持喪失時の即時消去、時刻の非更新、
支持回復時の再集計待ち、世代退役・別bus／epochの隔離を確認した。
cacheを読む前後でphrase snapshotが変化しないことも検査した。

通常binaryの既存prototype検査にacoustic／memory／gesture／period／phraseを組み合わせた。
presentationの実音から共有groupとprivate-audio prototypeの正の対応を確認し、
無音habitatへの対応漏れ、bus／epoch／モデル版、窓・利用可能時刻、100 ms間隔を検査した。
off／observe×report有無の音声byte一致を維持した。

次工程は、選択groupのfeatureに対するobserved／hypothetical／unsupportedの射影規則と
実身体の支持範囲を固定し、7 action×32 cellの帰結表を作ること。
共有descriptorへの近さを、未測定の候補envelopeや未来のaccent／retrieval支持に置き換えない。
その後のactual-Voice射影、default差分、私有trace・own-excluded energy列、通常report／UIの
候補診断、4／16／64 Voiceの全I10資源測定を含む終了条件は維持する。

最終検査: 全Rust 1,066成功、0失敗、30ignore。
`cargo test exit=0 @ 2026-09-18T00:12:50+09:00`。全出力と終了コードを同一shellで保存した。
通常Clippy、全target check、format、diff検査も通過した。
次の順でパス・NUL・内容・NULを連結したSHA-256は
`1161b6f6675001e7c93d8dc104c35c2ca738d7d67953cb6aa8e21b6ba0dd5dd0`。

- `src/temporal_cognition/body_model.rs`
- `src/temporal_cognition/body.rs`
- `src/temporal_cognition/observables.rs`
- `src/temporal_cognition/phrase.rs`
- `src/temporal_cognition/phrase/tests.rs`
- `src/temporal_cognition/observation.rs`
- `src/temporal_cognition/observation/replay.rs`
- `src/runtime/mod.rs`
- `src/ui/windows.rs`
- `tests/render_binary.rs`

### 実renderer envelopeの発行時固定（実装前登録、2026-09-18）

候補射影の入力として、Toneのattack／decay／sustain／releaseと絶対sample時刻を
コピー可能な読み取りモデルへ取り出す。実rendererと同じgain計算を使い、発行済みの
onset／release予測へその時点のenvelopeを保存する。release候補は元のhold終了より
早い場合だけ終了を短縮し、既存tailを延長しない。後のTone更新や結果学習で保存値を変えない。

このgainは合成器出力へ掛ける無次元envelopeであり、実音energy、modulator、
backendの共鳴・減衰、追加impulse、将来の指令を含まない。過去2秒の記述値から
これらを補わず、envelope単体を候補帰結表や7 class全体の完成とはしない。
原gain式との全sample照合、早いreleaseと長いtail、無期限sustain、飽和時刻、
発行後の不変性、通常binaryのreport接続と音声非干渉を検査する。

### 実renderer envelopeの発行時予測への接続結果

`life::sound::envelope::Envelope`へ従来のgain式と8個の状態値を移し、Toneはその値を
保持して実renderに使う。`prediction_parameters`も同じ値をコピーし、releaseを仮に
適用したコピーを返す。実Tone、乱数、backend、modulator、指令queueは変更しない。
`body::Capture::prediction_input`から自声予測へ渡し、`self_sound_outcome.prediction.envelope`
に発行時の絶対tickと形状を公開する。従来の終端スカラー予測はこのenvelopeの終端を使う。
RMSの固定近似や学習式、教師値の観測条件は変更していない。

早いreleaseではhold期間に合わせてattackも短縮する既存のgain挙動を保存した。
releaseを後から遅らせてもtailを延長しない。無期限sustain、u64終端の飽和、
発行後のnote_on／note_offと学習による保存値の非変更を検査した。
追加経路は固定サイズの値コピーであり、hop内の新しいheap確保や音声合成を行わない。
全I10の費用はまだ測定していないため、この性質を資源合格へ読み替えない。

通常binaryの既存検査にonsetとreleaseのenvelopeを確認するassertionを加えた。
480 sampleのattack、14,400 sampleのrelease、発行済みonsetの無期限holdと
後のrelease終端の違いを実reportから確認し、off／observe×report有無の音声一致を維持した。

保存済み36素材を変更後rendererで再生し、WAV全件byte一致、公開descriptor 1,512件の
独立照合も通過した。適合1,272、未知240は変更前と同じ。原reportと比較結果は
`target/i10-envelope-replay-20260918/`に保存した。再実行コマンド:

```sh
python3 scripts/verify_temporal_body_prototypes.py \
  target/i10-body-prototypes-20260917/corpus.json \
  target/i10-body-prototypes-20260917/model-8.json \
  --output target/i10-envelope-replay-NEW
```

さらにsine／harmonic／modalの各身体をADSR=(0.1,0.2,0.3,0.8)で鳴らし、
0.04／0.2／0.8秒後にreleaseする9条件を変更前binaryと比較した。
attack中／decay中／sustain中のreleaseを含め、全PCMとWAVがbyte一致した。
各条件の`.rhai`、両binaryのreport・WAV・log、hash付き`summary.json`は
`target/i10-envelope-boundaries-20260918/`に保存した。binary SHA-256は
変更前`c53348bdd04737dd47c557294749c4ea09e9f5669ba8bad7ddf03d907376b27d`、
変更後`fa27c3695dd4ea48e522dcfac7225e644860607b4b0d2ba43d5267d1a61986a5`。
この比較は既存音声の保存を示す。未来のacoustic energyや候補順位の精度は示さない。

次は同じ身体recipeでの7 classの射影とprototype側の支持範囲を登録し、
候補featureのobserved／hypothetical／unsupported分類、帰結表、実Voiceのdefault差分へ
接続する。今回保存したenvelopeは発行された一Toneの状態であり、Voice全体の重畳音、
modulatorやbackendを含む未来energy、全classの合法性、共有prototypeの候補モデルではない。
私有trace・own-excluded energy列、両busのreport／UI、4／16／64 Voiceの全I10資源検査も
未達のまま維持する。I9は再開していない。

最終検査: 全Rust 1,068成功、0失敗、30ignore。
`cargo test exit=0 @ 2026-09-18T00:28:50+09:00`。全出力と終了コードを同一shellで保存した。
通常Clippy、全target check、format、diff検査も通過した。
次の順でパス・NUL・内容・NULを連結したSHA-256は
`2be9d5536a99c4be4b6fc8a1a7c9323d353a67ba035f1ccbaac19d2bc9323a05`。

- `src/life/sound/envelope.rs`
- `src/life/sound/mod.rs`
- `src/life/sound/tone.rs`
- `src/life/self_prediction.rs`
- `src/life/self_prediction/tests.rs`
- `src/temporal_cognition/body.rs`
- `src/temporal_cognition/body/tests.rs`
- `tests/render_binary.rs`

### 7 classの実身体反実仮想素材（実装前登録、2026-09-18）

候補射影の比較先として、既存のoffline `ScheduleRenderer::fork_source`を使い、
同じ発行時身体から7 classの実PCMを作る。新しいinstrument経路やオンライン学習は加えない。
条件は48 kHz、480 sample hop、固定source ID、sine／harmonic／modal、
both／habitat／presentation routing、既存音なし／二Tone重畳中、SeqGate／DroneSway。
全条件を列挙し、失敗・無音・不適格classを後から除外しない。

判断時刻は0.3秒、継続は4秒。候補時刻は0／0.125／0.5秒後。
onset-now・skip・continueは0、delayed-onset・waitは正の二時刻、release・gapは三時刻。
continueとreleaseは既存身体がその時刻に活動する場合だけ適格。gapは既存Toneをreleaseし、
判断時に固定した0.2秒間は新しい励起を追加しない。wait／skip／continueは同じ自然継続を
使うが、待機・機会消費・継続という予定上の意味を別に記録する。release tailを無音にしない。
既存の二Toneと追加onsetのrecipe、amp、pitch、modulator、更新指令を全候補で固定する。

他Voiceを含むrendererから抽出した自身体snapshotを、他Voiceを含めず最初から独立再生した
同じ身体と全sample照合する。未知の外部未来を自声の予測へ入れない。
両busのPCMと10 ms窓の実energy、class・時刻・固定recipe・適否・withholding終端を保存する。
これは合成器状態とclassの意味を検査するground truthであり、onlineの候補feature／帰結表、
ordinal校正、候補の音楽的採用、R2資源合格を代用しない。

### 7 classの実身体反実仮想素材の取得・独立検証結果

`examples/temporal_action_profiles.rs`を追加し、36身体・routing・活動状態・modulator条件を
全列挙した。各条件13個のclass／時刻を登録し、396分岐を実renderした。
音のない身体のcontinue／releaseの72分岐も不適格として残した。分母は468分岐で固定する。
現在dueの機会とrate許可はこのassayで与えた条件であり、実Voiceの生態的な合法性判定ではない。

0.3秒の同じ実音prefixから4秒のPCMを生成した。他Voiceを含むrendererから抽出した
自身体snapshotと、他Voiceなしでprefixから独立再生した身体は、396分岐すべてで両busの
全sampleが一致した。prefix中に発行し将来に予約したpitch／amp更新も同じsnapshotに保持した。
両者に同じ将来rhythmを明示入力し、他Voiceの未来指令を予測へ渡していない。

wait／skip／continueの自然継続は一致し、delayed-onsetは励起前のPCMを変えなかった。
onsetは自然継続と異なるPCMになった。活動中のrelease／gapの108分岐では、
判断時に固定した1周期の区間にもtailが残り、自然継続とは異なるPCMになった。
gapとreleaseのPCMは同じだった。今回の素材はどちらも後続の新規励起を予定しないため、
この一致からgap終端での再許可やwait／skipのlive scheduler実装を合格にはしない。
それらの意味はmetadataとして保持し、後続の実消費者で別に検証する。

`scripts/verify_temporal_action_profiles.py`は保存された生PCMを読み直し、10 ms窓の
energyをPythonの`math.fsum`で独立計算した。316,800座標が許容誤差内で一致し、
最大絶対差は3.642919299551295e-17だった。素材集合、class／時刻の列挙、routing、
既知無音、tail、遅延onset前の一致、機会消費とwithholding終端を検査した。
素材欠落、energy改変、gap終端改変、不適格判定の改変、PCM改変の5負例はすべて拒否した。

原PCM、profiles、manifest、検証結果、負例結果は
`target/i10-action-profiles-20260918-v2/`に保存した。最初のmetadata追加前runも
`target/i10-action-profiles-20260918/`へ保持するが、正本はv2とする。
条件・433ファイルのSHA-256・generatorとverifierのhash・再実行手順・未評価範囲を
[I10 action profile記録](i10-action-profiles.json)へ保存した。
同記録のSHA-256は`ef845ba7045762e4f2abbbdaf46891cbde0516ae780c882fc4acf2689e5a603c`。

```sh
cargo run --example temporal_action_profiles -- target/i10-action-profiles-NEW
python3 scripts/verify_temporal_action_profiles.py target/i10-action-profiles-NEW
```

これは将来の候補射影を評価するdevelopment素材であり、未評価の射影を合格にしたものではない。
既存8 medoidモデルは変更していない。通常workerの候補feature・7×32 cellの帰結表、
actual-Voiceの射影範囲とdefault差分、私有trace・own-excluded energy列、
両busの候補report／UI、4／16／64 Voiceの全I10資源検査は引き続き必須の未達工程。
次はこの実PCMとの比較を使い、候補featureの支持と近似範囲を具体化して通常表へ接続する。

最終検査: 全Rust 1,068成功、0失敗、30ignore。
`cargo test exit=0 @ 2026-09-18T00:42:05+09:00`。全出力と終了コードを同一shellで保存した。
通常Clippy、新exampleのClippy、全target check、format、diff検査も通過した。
この工程でproductionのsrcは変更していない。新exampleはoffline専用であり、
conchordal instrumentへ音声書き出しや反実仮想renderを追加していない。

### 私有traceの時刻候補差分（実装前登録、2026-09-18）

発行時に固定したreference・anchor・私有trace分布から、仮の自声帰結時刻と身体defaultの
時刻のfit差分を読み取る。各headの正規化には非周期overflowも含め、内部binを再正規化しない。
32 binの中心間を線形補間し、anchorの区間一様分布上で積分する。周期referenceはwrapし、
非周期referenceは候補・default両方が[0,4]周期内に入るanchor部分だけを共通支持にする。
重みは元のreference×anchor重みを保ち、欠けた部分を他へ配らない。初期一様priorは
従来の結果採点専用とし、このfitの支持には使わない。

通常接続では、実onset／releaseの発行時に得た身体予測を使う。onsetの基準は既存固定予測と
同じ予定onset+1 sample、releaseの基準は同じ実envelopeの予測終端。これは予測時刻であり、
実際に聞こえた時刻へ書き換えない。判断時刻からmin(2×intrinsic period,4秒)の12等間隔点と
基準時刻を候補帰結時刻として使い、重複を除く。period不明時は基準点だけにする。
これはclassの合法性や候補PCMを推定する処理ではなく、後続class射影からも呼べる時刻fitの
診断経路である。新しい生成作用は加えない。

同じVoiceで50 ms未満の再計算は省略し、発行時の身体世代・基準モデル・支持時刻を記録する。
私有traceの教師値は引き続き実音の観測結果だけとし、候補読み取りはtrace・reference・
記憶を変更しない。周期wrap、anchor幅、非周期overflow、両向きの片側支持、部分共通支持、
未知trace、世代交代、rate制限、発行後不変性、実binaryのreport非干渉を検査する。

### 私有traceの候補／default時刻fitの通常接続結果

`private_trace::paired_lookup`が候補とdefaultに同じanchor支持を適用する。
周期は32 bin中心間の補間を区間一様anchor上で積分し、完全な周期は平均massとしてまとめる。
非周期はanchor区間を両時刻が[0,4]周期内に入る共通部分へ切り詰め、その部分の元の重みを
使う。正規化済みheadにはoverflowのmassも残し、内部32 binへ配り直さない。
内部massがゼロでも共通支持があれば既知ゼロとして返す。

`generate_temporal_private_fit_fixtures.py`はFractionの有理数演算と区分線形関数の
hinge基底の原始関数で64条件を独立計算した。Rust側の区間分割・台形和とは別の計算である。
両向きの片側支持、部分共通支持、periodic wrap、完全周期、点anchor、狭い区間、
遠い時刻、overflowだけの既知ゼロを含み、絶対誤差2e-11未満で一致した。
最初の検査では遠い時刻からanchorを引いてから周期へ割ると位相を失うことを検出した。
周期剰余を引き算の前に取り、短い区間の幅も位相とは別に保持する形へ修正した。
fixtureのSHA-256は`0cca04fca82eb929f717d835df205578732cafec12869ead5bca446fc5dba449`。

通常の`Observer::predict`から、発行時に固定済みのreferenceと両traceモデルの分布を読む。
初期一様priorは使わない。onsetの基準時刻は既存固定近似の予定時刻+1 sample、releaseは
発行時envelopeの終端とし、実観測時刻へ後から置き換えない。
12個の等間隔帰結時刻とこの基準を重複除去し、最大13点を固定配列で評価する。
intrinsic periodが不明なら基準点だけにする。Voiceごとに50 ms未満の再計算を省略する。

`private_participation_trace.timing_preview` version 1へ、身体世代・基準モデル・基準時刻と、
各時刻の候補mass、共通支持上のdefault mass、差分、共通支持、未割当massを保存する。
fitsの順序はfull trace、elapsed-only control。reference重みとanchor重みを保ち、
未知・範囲外のmassを他へ配らない。双方の共通支持がゼロなら数値差分をnullとする。
既存recordのepoch、bus、発行時刻、inventory時刻とともに通常reportへ届く。
UIにもpreview数・query数・rate制限数を表示する。候補を`observe`へ渡す経路は追加していない。

単体検査で、未知traceのprior非使用、照会前後の学習分布不変、照会の重複抑制、
後のcontext変更と学習による保存値の非変更、50 ms境界、period不明時の一点、
release終端、Voice世代交代、habitatへroutingしない場合の非適用を確認した。
通常binaryの既存の長い実音fixtureにもbody予測を組み合わせ、学習済みの同じVoiceから
非ゼロの候補／default時刻差分がreportに現れること、default自身の差分がゼロになること、
未割当massが保持されることを確認した。off／observe×report有無のWAV byte一致を維持した。

この経路は実行済み指令の身体予測を基準にした帰結時刻fitの診断であり、
7 classごとの実身体射影、行動の合法性判定、未実行候補の音響モデルを完成させたものではない。
共有候補feature・帰結表、classごとのactual-Voice射影と全列のdefault差分、
own-excluded energy列、両busの全候補診断、4／16／64 Voiceの全I10資源検査は未達として残す。
新しい生成作用は無効、I9保留も維持する。

最終検査: 全Rust 1,071成功、0失敗、30ignore。
`cargo test exit=0 @ 2026-09-18T01:03:26+09:00`。全出力と終了コードを同一shellで保存した。
通常Clippy、全target check、format、diff検査も通過した。
次の順でパス・NUL・内容・NULを連結したSHA-256は
`9e9d2ee1f2832321df13d0117101297150385053e1c2fd6c9b8bf2b63ef1883b`。

- `src/temporal_cognition/private_trace.rs`
- `src/temporal_cognition/private_trace/tests.rs`
- `src/life/participation_trace.rs`
- `src/life/participation_trace/tests.rs`
- `src/life/action_observation.rs`
- `src/ui/windows.rs`
- `tests/render_binary.rs`
- `scripts/generate_temporal_private_fit_fixtures.py`
- `tests/fixtures/temporal_cognition/private_fit.json`

### 外部energy窓の支持と16点照会（実装前登録、2026-09-18）

I10のenergy列は窓中心間の補間を要求するため、既存の局所own-excluded forecastへ
読み取り専用の窓照会を追加する。既存のband_energy_at／energy_window_afterと
行動選択の計算は変更しない。元の10 ms予測窓の値を3帯域の単純和で読む。
これは既存filterbankのenergy推定であり、raw PCMのmean-squareそのものとは呼ばない。
追加の帯域重みやshared tableのenergyは使わない。

隣接する支持窓の中心間だけ線形補間する。隣が未知または範囲外なら現在の支持窓の端まで
その窓の値を保持し、未知窓へは入らない。時刻は整数sampleと小数部で受け取り、
大きな絶対時刻へ小数を足して精度を失わない。非有限・負の予測値は未知とする。

既存の局所参加予測の診断へ、予定onsetから既知hold＋releaseまでの要求区間、
4秒上限と外部予測の支持端で切り詰めた区間、除外した前後のsample数、
その区間の16等間隔中点における外部energyを記録する。除外した長さをenergy massとは
呼ばず、自声の予測がない状態でoverlap／audibilityを計算しない。
この入力照会はreportの有無によらず同じであり、選択costや学習へ追加しない。
境界、半sample、欠測窓、既知無音、4秒clip、長いtail、8／16／32点の基準値を検査する。


### 外部energy窓の支持と16点照会の接続結果（2026-09-18）

`TemporalForecast::centered_energy` は既存の3帯域予測を窓中心間で補間する。
元の `band_energy_at`／`energy_window_after`／`sustained_energy_at` の計算は維持した。
値は既存filterbankの3帯域energyの単純和であり、raw PCMのmean-squareと同一とはしない。
非有限・負の窓は未知とし、隣接支持窓がなければ現在の窓端まで値を保持する。
欠測窓をまたぐ補間、予測終端を越す背景energyへの外挿は行わない。

`external_footprint` は要求区間、外部予測の外枠との `horizon_intersection`、
前後に除いたsample数、16等間隔中点を固定長で返す。内部の欠測は各点の
`band_energy_sum: null` に残す。空の交差では区間・前後の除外数もnullとし、
未算定を無音またはtail全欠落へ読み替えない。整数sampleと小数部を別々に保持し、
u64の大きな絶対時刻でも半sample以下の位置を失わない。

既存の `TemporalParticipation` が選んだ予定についてだけ、予定onsetから
hold＋releaseまでの区間を、forecastのissue基準4秒上限で切り詰めて記録する。
探索中の全候補への重複計算は避けた。既存のplanned／pending contextとともに
発行時点で凍結し、通常 `participation_context` reportへ渡す。
reportの有無でこの処理を変えず、選択cost・既存context学習へ値を加えない。
これは既存参加診断への追加であり、offで新しいrelation worker／bankを起動しない。
配列は固定長で、追加のhop内heap確保はない。コピー量・CPU時間を含む全費用は
後続の全I10資源測定に含める。

追加した5数値検査は、窓中心・左右端・小数sample・未来支持外・非有限入力、
部分観測窓、NaN／負値の内部欠測、既知無音、4秒clip、空区間、長い要求tail、
巨大な時計原点、8／16／32中点のaffine基準値を検査した。
既存の実音検査にも16点診断を追加した。自声と隣声が逆位相で打ち消し合う場合、
shared混合予測はゼロだが、局所own-excluded予測の16点は正の隣声energyを保持する。
発行済みcontextは後続の実観測学習でも同じ16点を保持する。
通常rendererの既存off／observe×report検査は、reportの16点・時刻・上限を検査し、
モードとreport有無によるWAV不変性も通過した。

別途、保存済み `/tmp/conchordal-render-before-i10-envelope` と現行debug rendererを、
同一seedのentrained Voice＋隣声素材でoff／observe比較した。旧／新×2 modeの
4 WAVは全byte一致し、SHA-256は
`9a99ae2e73f6dd485f7e76d07fa394c6d892d39bd7e0d024e5b4008e77c4e6b3`。
現行reportには各modeで4件のfootprintを確認した。
素材、config、report、WAV、比較結果は `target/i10-external-energy-20260918/` に保存した。
この比較は当該固定素材の非干渉検査であり、全身体の予測精度の証拠ではない。

最終検証は `cargo test -- --nocapture` が1,076成功、失敗0、無視30（31 suite）。
`test_status.txt` は `cargo test exit=0 @ 2026-09-18T01:25:09+09:00`。
通常Clippy、全target check、fmt、diff checkも通過した。
検証対象の順序付き `path + NUL + content + NUL` SHA-256は
`5ed8100725acbfb411dfea61ef0fc116e192f6a9f600e662dc61afcfad1e20d5`。
対象順は `src/core/temporal_expectation.rs`、`src/life/temporal_participation.rs`、
`tests/render_binary.rs`。

この16点は外部energyの入力診断であり、自声の実身体energy profile、
overlap／audibility比率、7 classの候補帰結表の代用ではない。
次は自声energyの支持・単位を対応させ、候補feature射影とshared table／actual-Voice差分、
両busの全候補report／UIを接続する。4／16／64 Voiceの全I10資源検査も未達。
I10の完了条件は縮小せず、I9保留と新しい生成作用の無効を維持する。


### 実音energyの16点近似監査（実装前登録、2026-09-18）

既存の局所自声profileはrouting前の減衰peakであり、未観測ゼロ・body世代・未来の
実envelopeとの対応を持たない。そのままI10の自声energy予測として接続しない。
先に固定済み396 branchの両busについて、保存PCMから検証済みの10 ms energy窓を
理想的に既知の自声energy曲線として使い、数値積分の誤差を測定する。
これは未来PCMを使うoffline fidelity targetであり、ライブ予測器や観測学習へ投入しない。

追加する監査では8／16／32等間隔中点を、窓中心・端で区切った区分Gauss–Legendre
積分と比較する。dense側は8／16次数を照合し、比率差と積分誤差を記録する。
自声の要求区間は登録済みのonset・hold・releaseから作り、既存toneと追加toneの
supportの和集合の外枠とする。内部の音間gapも残す。無音branchは4秒までの既知無音を
対象とする。外部energyは独立に宣言した無音・定数・ramp・短いburst・内部欠測・
短い支持終端を使う。共有混合energyを自声除外energyとして扱わない。

同じ点・同じ正の自声積分を使い、epsilon=1e-12でoverlapとaudibilityを計算する。
ゼロ自声積分は、その区間全体で無音が既知の場合だけoverlap=0、audibility=null。
16点が短い非ゼロ音を見落とした場合は無音と認定しない。支持の内部欠測も点の標本だけで
推定せず、窓の支持区間から検査し、欠測をまたぐ条件の比率は未知にする。
clipした自声energy積分と質量比を別に記録する。

全case・両bus・適格候補を分母に保持し、body／modulator／class別の絶対誤差と、
同じcase・bus・外部条件での候補順位反転を記録する。差を隠す許容tieは導入しない。
元々同一の無音・release/gap等は厳密tieとして別集計する。合格閾値を後付けせず、
この監査だけでは予測精度・可聴性・ordinal校正・I10全体を合格としない。


### 実音energy積分監査と区間支持の接続結果（2026-09-18）

独立監査 `scripts/evaluate_temporal_energy_footprints.py` を追加した。
入力433ファイルのhashを検査し、verification自体も固定済み
`i10-action-profiles.json` の記録と一致させる。検証JSONだけの差し替えでも
再固定なしの別素材を受け入れない。元の396適格branch、両bus、8外部条件を保持し、
6,336条件を比較した。72不適格branchは取得記録のまま分母として別記した。

基準は10 ms実音energy窓の中心／端で区切ったGauss–Legendre積分。
初期8／16次数で最大6.53e-5の比率差が残ったため、基準側だけ次数を倍増し、
前次数との比率変化が1e-10以下になるまで最大512次数で照合した。
最終の最大変化は8.74e-11、未収束条件0。これは数値収束検査であり、厳密な誤差上界の
証明とは区別する。ライブの16点上限や採用閾値は変更していない。

16点の状態はsupported 2,688、known_silent 3,252、unsupported 360、
empty_intersection 36。known_silentはoverlap=0・audibility=nullとして扱うため、
overlapの比較分母は5,940、audibilityは2,688となる。
自声の区間積分が標本上ゼロでも、区間全体の無音が既知でなければ無音としない。
短い非ゼロpulseを全標本が見落とす反例もPython検査に含めた。

| 中点数 | overlap最大絶対誤差 | audibility最大絶対誤差 | overlapのp95誤差 | audibilityのp95誤差 |
|---|---:|---:|---:|---:|
| 8 | 0.843878 | 0.843878 | 0.136593 | 0.274331 |
| 16 | 0.239897 | 0.239897 | 0.051759 | 0.080603 |
| 32 | 0.117328 | 0.117328 | 0.030464 | 0.049340 |

16点での各energy列の順位比較は32,832組ずつ。overlapは2,520組がunsupported、
残る30,312組で厳密な符号反転644組、dense側の厳密tie 17,544組。
audibilityは19,272組がunsupported、残る13,560組で反転896組、dense tie 1,512組。
微小差を勝手にtieへ丸めず、基準差が1e-3より大きい反転も別集計した
（overlap 496組、audibility 604組）。これは列ごとの順位であり、全帰結のutilityや
最終行動順位ではない。body／modulator／class別の全誤差と候補組比較も保存した。

最大の16点誤差はmodal・two_tones・swayの0.5秒後gapと、issue後120–140 msの
短い外部burstの組み合わせ。overlapは基準0.100978に対して16点0.340875。
同じactive-bodyのcontinueでは16点がburstを見落とし、基準0.097023に対して0となる。
この監査は未来の実音energyが理想的に既知でも積分で差が生じることを示す。
自声予測モデルの誤差を既に含む、またはlive予測が正しい、とは解釈しない。
生成作用への昇格条件は未成立のままとし、16点参照のdiagnosticとdense fidelity targetを
別に保持する。事後的な受入閾値や、結果の悪い身体の除外は行っていない。

支持については、16点がすべて既知でも点間に未知窓が残る反例を確認した。
`TemporalForecast` に7語・56 byteの支持bitsetを追加し、共有／局所energy予測の
更新時に一度だけ構成する。各10 ms窓の3帯域の有限・非負条件と、粗い予測行間の
補間に必要な両端の既知性を反映する。`external_footprint` は最大7語のmask比較で
`continuous_support` を返す。空交差はnull、内部欠測はfalse、全窓支持はtrue。
16点それぞれの値とは別に報告し、比率消費側が点だけから完全支持を推定しない契約とした。
追加のheap確保や候補ごとの全予測窓scanはない。予測更新・コピーの費用は全I10測定へ含める。

Rust検査は標本間の穴、bitsetのword境界、支持端、通常renderer reportのtrueを追加した。
Pythonは定数比率、中心・端・欠測、小数sample、既知無音、短いpulse、
解析的なlog積分との一致、負energy、空交差、入力改変とverification差し替え拒否の8件。
全Rustは1,077成功、失敗0、無視30（31 suite）。`test_status.txt` は
`cargo test exit=0 @ 2026-09-18T01:39:51+09:00`。通常Clippy、全target check、fmt、
diff checkも通過した。

正本の機械記録は [energy積分監査](i10-energy-quadrature.json)。
全6,336行と順位比較は `target/i10-energy-quadrature-20260918-final/` に保存した。
それ以前の同名prefixディレクトリは数値基準を細かくする前の試行であり、最終結果と混同しない。
変更した4ファイルの順序付き `path + NUL + content + NUL` SHA-256は
`c3b3b1d872a71f2d1f3097478f33e60cdedbbf81ad797b33d031742bea67bacf`。
順序は `src/core/temporal_expectation.rs`、`tests/render_binary.rs`、
`scripts/evaluate_temporal_energy_footprints.py`、`tests/test_temporal_energy_footprints.py`。

I10全体は未完了。今回、ライブ自声energy予測と比率の接続は完了していない。
次は私有のbody世代・routing・実envelopeと整合する自声energy予測を固定し、
上の実音targetと比較する。既存のrouting前peak profileをその代用としない。
候補feature射影、shared帰結表、actual-Voice／default差分、両busの候補report／UI、
4／16／64 Voiceの全I10資源測定も引き続き必須。I9の保留は維持する。


### 私有source-energy予測（実装前登録、2026-09-18）

指令単位の20 ms onset／2 s release診断を入口に、同じVoiceの全toneを加算した後の
私有routing済みPCMを16時間窓へ集計する。tone energyの和は使わず、同じsource内の
干渉を含める。既存body captureのframeを送信前に読む。source／source generation／
body generation／capture slotを一致させ、未取得・非有限・別世代のsampleは支持に数えない。
各窓の全sampleが揃った座標だけを教師にし、終了時の未完窓は未知にする。
新たなcapture buffer、音声出力先、spectral解析は追加しない。

窓境界は start+ceil(k*width/16), k=0..16。空窓は未知。
出力は線形full-scale-squaredの窓平均energy。既存自声予測に平行な16出力を追加し、
実身体envelopeの窓中点gainと指令amplitudeによる amp^2*gain^2/2 を固定sine対照にする。
これは全backend・modulator・既存toneを正確に再現する物理モデルではない。
動作条件付きモデルはこの対照へのresidual、動作非依存対照はenergy=1e-4へのresidual。
既存18特徴に当該busへの指令routing flagを加え、非依存対照では指令特徴1..5とroutingを
ゼロにする。単位変換 log1p(E/1e-6) 上で、既存と同じ正規化LMS
w += 0.1*(target-issued_prediction)*x/(1+sum(x^2)), weight clamp [-8,8] を使う。
予測logは0からfinite f32最大値の二乗に対応するlogまでに制限する。
両bus・両学習対照・16窓のweightsはsource/body世代ごとの既存64枠内に保持する。

学習前に、発行済み固定／条件付き／非依存予測を、線形energyの二乗誤差とlog単位の
二乗誤差で記録する。後から更新された予測へ差し替えない。未実行候補・取消・欠測は
学習させず、世代退役後の結果は記録しても別世代を更新しない。
予測は指令診断窓内だけを支持し、それ以降のenergy／tail massは未知のままとする。
指令toneのenvelopeは既存の凍結値を保持するが、source全体の無音証明へ使わない。

まず通常renderer／report／UIの予測・教師・誤差を接続する。実PCMとの独立な窓集計、
両bus、同一sourceの干渉、支持欠落、世代・発行時凍結、学習するゼロ、音声不変性を検査する。
これは7 classの反実仮想モデルや全Voice帰結表の完成ではなく、その実音予測・検証経路。
前段で確認した16点積分誤差の監査を代用せず、生成作用は無効のままとする。


### 私有source-energy予測の通常接続結果（2026-09-18）

`source_energy_log1p_residual_v1` を実装した。定義とコードhashは
[source-energyモデル記録](i10-source-energy-model.json)に保存した。
`Forecast.source_energy` はhabitat／presentation順の2laneを持ち、16窓の予測値・
教師・線形energy二乗誤差・発行時更新数を通常 `self_sound_outcome` reportに記録する。
集計ではlog単位の二乗誤差も保持する。UIへscored windows／updated／unsupported／retiredを追加した。
既存の指令toneの5座標予測と、source全体のenergy教師を別フィールドにした。
指令が片busへ送られていなくても、そのsourceの他toneが寄与し得るため、
source-energy側は両busを保持し、実際のcaptureで無音を確かめた窓も学習する。

教師は `Capture::source_audio` から、送信前の私有PCMを読み取る。
source ID・source generation・body generation・slotを照合し、同じVoice内の全toneを
足した後のwaveformを二乗する。`Observer::observe_source_energy` は各窓の有限sample数を
数え、全sampleが揃った窓だけを平均energyにする。欠測hop・非有限sample・body世代変更は
対応窓を未知にし、窓間で支持を埋めない。performance終了などで2秒のrelease診断全体が
揃わなくても、完了した個別窓は教師になり、未完窓はnullに残る。取消・superseded指令は
個別窓が取得済みでも学習から除外する。

モデルは指令発行時に固定／条件付き／動作非依存予測を凍結し、完了時にそれらを採点してから
正規化LMSを更新する。body世代の退役後に届いた教師は発行済み予測と照合できるが、
置換後のモデルへは学習させない。予測窓はonsetの20 ms、releaseの2秒に限定し、
その先の詳細energy、source全体の終端、tail massはこのモデルから捏造しない。
固定対照の実tone envelopeは近似の基底であり、複数toneを含むsource全体の無音証明ではない。

新規4検査を追加した。
発行後に別結果が学習されても元予測を維持すること、body世代退役時の非更新、
3 sampleを16区間へ分けた空窓、学習する既知ゼロ、取消の非学習、
動作非依存対照の入力隔離、動作対応を反転した場合の再学習を検査した。
合成対応検査では各対応1,600試行の後半200試行について、条件付きモデルの平均log二乗誤差が
0.05未満、固定・動作非依存対照がともに1を超えることを確認した。
これは学習機構の対応検査であり、実音での予測精度や可聴性の受入ではない。

private captureを通す検査では、時間変化する同一sourceの2波形を同相／逆相で足し、
片方だけをpresentationへ送った。16窓ごとの独立なwaveform集計と教師が一致した。
片busのhop欠落、単一の非有限sample、異なるbody generationも検査した。
逆相ではpresentationのsource-energyがゼロでも、個々のtoneのRMSは正のままとなる。
この差を保持することで、tone energyの和に置き換わっていないことを確認した。

通常rendererの検査は、onsetの16×60 sampleの教師積分が同一sourceの独立な全窓RMS二乗に
一致すること、routingで離れたbusの既知無音、発行時予測から計算した誤差、releaseの
取得済み窓と未取得窓の分離を確認した。release検査はscenario終了で指令がsupersededに
ならないよう待機を2.1秒に延ばした。off／observe×report有無でWAVは全byte一致した。

追加のmodel weightsは64 owner×2 bus×2学習対照×16窓×19特徴×8 byte = 622,592 byte。
256 pending commandのsample数／energy積算配列は計131,072 byte。
これは追加配列の算術上の容量であり、拡大したForecast／Outcomeのコピー・report・UIや
総ModelBankの常駐量、CPU費用を含む資源検査ではない。窓集計は既存frameを読むため
追加のPCM ring・spectral解析・hop内heap確保はないが、指令ごとの集計費用は
全I10の4／16／64 Voice測定へ必ず加える。

最終の全Rustは1,081成功、失敗0、無視30（31 suite）。
`test_status.txt` は `cargo test exit=0 @ 2026-09-18T02:00:54+09:00`。
通常Clippy、全target check、fmt、diff checkも通過した。
順序付き `path + NUL + content + NUL` SHA-256は
`db9eeb2624c0584be0a54a6099aac8cb0abfcda46c9b718d8098c535de43ff50`。
対象順はモデル記録の `source_sha256` の挿入順と同じ9ファイル。

I10全体は未完了。次は、この予測を読み取り専用の候補射影で利用する際の支持・
未知・actual-Voice/default対応と、外部energyとの同じ絶対時間上の比率診断を接続する。
7 classの候補feature・shared帰結表、両busの全候補report／UI、全資源測定も必須のまま。
新しい生成作用は無効、I9保留も維持する。


### 指令発行時のenergy比率診断（実装前登録、2026-09-18）

既存の外部energy混合式から、発行記録・学習・scratch予測を書き換えないread-only照会を
分離する。I10のbody診断が有効な場合だけ、実onset／release指令を含むsourceについて
当該hopのhabitat own-excluded予測を保持する。64枠を初期化時に確保し、source generation・
body generation・issue tickを照合する。旧参加予測への余分なissue_comparisonは行わない。

指令発行時の条件付きwhole-source energy予測と、外部forecastの4秒上限・支持区間の交差を
16等間隔中点で読む。自声側も窓中心間の線形補間を使い、端窓内はその端まで保持する。
空窓・未知窓を越える外挿はしない。整数sample＋fractionを保持する。
外部のcontinuous_supportがfalseなら、16点が有限でも比率は未知にする。

同じ16点と重みで、epsilon=1e-12のoverlap／audibilityを計算する。
自声の予測積分がゼロでも、それだけでsource全体の無音とは認定せず、比率は未知とする。
外部予測が既知ゼロで自声予測が正ならoverlap=0を許す。
自声の窓内piecewise-linear曲線の積分を別に求め、外部支持との交差によって除外した
予測energy質量を報告する。指令診断窓以降の詳細energy／tail massは未知のまま。
自声はraw PCM energy予測、外部は既存3帯域energyの単純和という近似を明示する。

通常reportには発行時の点・比率・支持・モデル・原時刻を凍結して記録し、
後続の実音教師やモデル更新で置き換えない。UIには支持／未知の累計を渡す。
生成作用・新たな比較学習は無効。これは指令に対する診断であり、7 classの全候補表、
actual-Voice/default差分、全I10資源検査の代用ではない。

### 指令発行時のenergy比率診断の通常接続（2026-09-18）

`HistoryEnergyPrediction::preview_comparison` に既存の混合式を分離し、
`AcousticTemporalExpectation::preview_external_energy` から読み取る経路を接続した。
後者は実観測済みのrecurrence weightと当該sourceのown-excluded historyだけを読み、
旧参加予測の比較登録・学習・recurrence scratchを更新しない。
`advance_population` は実指令のbatchが確定した後、On／Offを含むsourceだけを照会する。
Updateだけのhopや候補ごとに外部予測を作り直さない。

Observerは初期化時に64 context分を確保する。source ID／source generation／body generation／
issue tickの一致を確認し、`Forecast.energy_ratio` へhabitatの16点・比率・支持を凍結する。
後のcontext交換、実音教師、モデル更新ではこの記録を書き換えない。
reportは通常の `self_sound_outcome.prediction.energy_ratio`、UIは同じ計算の発行・支持・未知累計を読む。
presentation側のwhole-source energy教師は維持するが、habitatの外部予測をpresentationの
own-excluded予測として転用しない。

自声側は条件付き16窓の中心間を線形補間する。32個の半窓ごとの中点積分で、
このpiecewise-linear予測曲線の全量と区間交差の量を別に計算する。
`modeled_own_integral_sample_units` は指令診断窓内の予測量であり、窓外のtail量ではない。
16点が短い予測energyをすべて見逃す場合は `missed_own_energy`、予測量自体がゼロなら
`zero_prediction_not_silence_proof` とし、どちらも比率をnullにする。
外部の既知ゼロと正の自声予測はoverlap=0として支持する。
外部は既存の3帯域energy和、自声はraw PCM energyの学習予測であり、尺度間の近似は未校正のまま。

新規5検査と既存の通常経路検査の拡張で、以下を検査した。

- 既存の状態変更あり混合とread-only混合の出力一致、450 stepにわたる比較学習・誤差の非干渉。
- 実波形の同相／逆相の自声除外での従来経路との一致、recurrence scratchと比較登録の非変更。
- 定数比率、外部支持と4秒上限による除外質量、`2^60+7` の絶対clock、奇数長・空窓、中心補間の積分。
- 16点の間にある自声／外部の欠測、予測ゼロと未知の分離、標本点が見逃す短い予測energy。
- source／source generation／body generation／発行時刻の不一致、context交換後の予測凍結、二重発行の防止。
- 通常binaryのpresentation専用sourceでのhabitat予測ゼロ、release時のread-only外部支持、
  off／observe×report有無のWAV全byte一致。

最初の全Rust実行では既存の `habituation_off_is_the_causal_control` が失敗した。
失敗するoff検査は設定ファイルを削除しない実装なのに、他の並列検査終了後にその設定ファイルが
残っていなかった。設定を含む一時pathが時刻だけの命名だったため、PIDとprocess内atomic連番を追加した。
初回の全出力と終了コードは `target/i10-command-energy-ratios-20260918/` の
`test-report-before-temp-path-fix.txt`／`test-status-before-temp-path-fix.txt` に保存した。
この修正は検査ファイルの分離だけであり、habituationの生成処理・判定値は変えていない。

I10全体の残要件は、7 classの候補feature射影と共有帰結表、actual-Voice／body-default対応と
paired差分、両busの全候補診断、4／16／64 Voiceの全費用測定・R2への引渡し。
今回の64枠の外部forecast cacheと比率計算も、全I10資源測定へ加える。
I9は保留を維持し、新しい生成作用は無効のままとする。

最終検査はRust 1,086成功、失敗0、無視30（31 suite）。
`cargo test exit=0 @ 2026-09-18T02:26:27+09:00`。通常Clippy、全target check、fmt、diff checkも通過した。
定義・12ファイルのhash・検査結果を[指令energy比率記録](i10-command-energy-ratios.json)へ保存した。
順序付き `path + NUL + content + NUL` SHA-256は
`48ef9fd3eb94ac508044e967f40b433667f72ba73a6dce1d0da26cc6c570baa5`。

### 観測／候補で共用する生特徴射影（実装前登録、2026-09-18）

候補射影の最初の接続点として、`features::Stream` の10個の生特徴の数値式を、
履歴・accent登録を持たない関数へ分離する。通常の観測Streamもこの関数を使う。
入力は当該区間のassigned energy／bus energy／Log2Space上のspectrumと、
必要な場合の直前区間の同じ入力である。純粋な数値関数は時刻・groupを推定せず、
観測側は従来の支持・隣接・association・grid検査を通した値だけを渡す。
候補側でも同じ物理hop・座標・尺度を満たした入力だけを使う必要があり、
16窓のsource-energy出力をそのままNSGT hop特徴として代入しない。

各出力はobserved／projected／unsupportedを区別する。単一frameの特徴はそのframeの
出所を引き継ぎ、隣接差分は現在と直前のどちらかが候補ならprojectedにする。
未指定のspectrum・energy・前区間は補わない。正のenergyに対してspectral massがゼロなら
fluxを未知にし、実際に支持された両区間のゼロspectrumならflux=0を許す。
centroid／spread／band fractionには正のspectral massを必要とする。

出力の順序・floor・式は現在のraw descriptorに固定する。accent検出用の別floorによる
flux補助値は維持するが、射影関数自身はaccent・observed_prefix・episode・creditを生成しない。
通常Streamが既存どおり実観測の検出・履歴更新を所有する。候補用の出所付き結果を
RawDescriptorへ暗黙変換する関数は設けない。

実装時に10特徴それぞれの投影方針をfeature manifestへ記録する。
同じ式の利用、観測のbyte相当出力、混合した出所、欠測・known zero・非有限入力、
支持外spectrum、Log2Space長不一致を検査する。これは共有帰結表への入力契約であり、
7 classの入力生成・head全特徴の投影方針・32 cell表・実Voice対応の完了ではない。

### 生特徴の共通数値経路と出所の分離（2026-09-18）

`feature_projection::evaluate` を通常コードへ追加し、`features::Stream::push` の
10個のraw descriptorとaccent用flux補助値を同じ関数から得るようにした。
元の観測時計・支持検査、隣接／association／gridの検査、accentの認定、観測数、履歴の更新は
引き続きStreamだけが行う。射影関数は入力への参照しか持たず、heap確保・観測状態の更新はない。
現在／直前のenergy、現在のbus energy、現在／直前のspectrumにそれぞれ出所を保持する。
出力は `observed`／`projected`／`unsupported` であり、数値が同じでも出所は混同しない。

feature manifestの `episode.raw.*` 10項目に、依存入力、支持条件、出所の伝播、観測への
非帰還を登録した。元の式・floor・順序・fit入力は変更していない。
この登録はraw層に限る。2秒／8秒head全入力の投影方針、issue coverageの固定、
7 classの身体入力と未来区間の支持は後続で接続する。

追加4検査では、五つの入力の出所を独立に切り替えた32組、既知ゼロ、spectrum欠落、
正のenergyに対してゼロspectral mass、非有限／負値、両spectrum境界の長不一致、
accent floorとdescriptor floorの分離を確認した。
従来の独立Python参照も1,152 endpoint／275 accentで一致した。

変更前binaryを保存し、sine／harmonic／modalと両bus／habitat専用／presentation専用を含む
同じ固定seedの実音を新旧binaryでrenderした。WAV全byte、384 body descriptor、
6 self-sound outcome、6 descriptor predictionが一致した。
共有reportは非同期snapshotなので、同じ出力行番号だけでは同じ原観測区間にならない。
bus・epoch・generation・原区間で照合した共通339件のraw descriptor／accent detectorと、
同じbus・epoch・support endの68件のphrase snapshotが一致した。
変更前binaryの再実行でも公開区間が変わり、共通331件のraw／detectorと67件のphraseは一致した。
共有report全行の同一性や、共通区間でない部分の一致は主張しない。

実音比較は `target/i10-feature-projection-20260918/`、定義・source／入力／binary／WAV／reportの
hashと検査数は[生特徴射影記録](i10-feature-projection.json)へ保存した。
3本のWAV SHA-256はすべて
`00ba126a393f45d4493fa56231229592d7010810ed7428c79b47c20853142a90`。
全Rustは1,090成功、失敗0、無視30（31 suite）。
`cargo test exit=0 @ 2026-09-18T02:39:19+09:00`。
通常Clippy、全target check、fmt、diff checkも通過した。
記録の5ファイルの順序付き `path + NUL + content + NUL` SHA-256は
`754aa70208aebef7bdab39cf42e2d683c1cffa47c371aa8eda64d5e7ac1af816`。

I10全体は未完了。純粋なraw数値経路を候補用にも使える状態にしたが、通常runtimeで
7 classの候補入力をこの関数へ流す消費者はまだない。次は身体prototypeの軌跡と同じ物理hopで
支持付き候補入力を作り、全headの出所・未知の方針を固定して共有帰結表へ接続する。
actual-Voice／body-default対応、paired差分、両busの候補診断、全I10資源検査も引き続き必要。
I9保留、新しい生成作用の無効、校正と作者採用の別判定は維持する。

### 物理hopと発行前PCMを保持する7 class素材（実装前登録、2026-09-18）

従来の7 class素材は480-sample hopの未来PCMだけを保存していた。通常の48 kHz／512 hopの
NSGT特徴を候補と実音で比較するには、同じhop境界と発行前のsource PCMも必要となる。
既存のv1素材は保持し、例の取得器へ480／512 hopの明示選択とprefix保存を追加する。
新schemaはv2とし、energyを10 msと呼ばず `energy_per_hop` へ記録する。

発行時刻は元の14,400 sampleを選んだhopの次の境界へ切り上げる。512では14,848 sample。
prefix内の第2toneは5,760 sampleのまま、その時刻を含むhopで予約する。
既知の更新は発行の2 hop前に渡し、変更時刻は発行後2,400 sampleのままとする。
未来4秒＝192,000 sampleは両hopで割り切れる。各caseのprefixをsource単独の独立replayから保存し、
全実行可能分岐で同じprefixを得たことを照合する。自声除外用の外部sourceはprefixへ混ぜない。

36 case／396実行可能分岐／72不適格の登録分母、3身体・routing・2tone／silence・modulator、
classと予約時刻の意味は維持する。各busのprefix／futureのPCMとhop energyを独立verifierで再計算する。
既存8 medoidのrecipe・周波数・発行時刻とは別条件なので、この素材をそのまま8 medoidの
行動templateとして登録しない。実身体の反実仮想targetであり、生成中の観測や学習教師でもない。

### 前史付き7 class素材の取得と独立検証（2026-09-18）

`temporal_action_profiles` 例へ `--hop-samples 480|512` を追加し、v2 schemaで
発行前の私有PCM、prefix／futureの物理支持、各hopのenergyを保存した。
source単独の独立prefix replayを各分岐で実行し、同一caseの前史の全byte一致を検査する。
未来のforkと独立replayの比較、release tail、wait／skip／continueの音響上の一致、
gapの固定withholding endpoint、未来の未予約excitationを追加しない条件は維持した。

取得先は以下の二つであり、旧素材は変更していない。

- `target/i10-action-profiles-512-prefix-20260918/`：通常設定と同じ512 hop。
  発行時刻14,848 sample、prefix 29 hop、未来375 hop。
  prefixの2,088 energy値と未来297,000値をPCMから独立に再計算した。
  最大絶対差は `3.2959746043559335e-17`。
- `target/i10-action-profiles-480-prefix-20260918/`：旧取得条件との比較用。
  発行時刻14,400 sample、prefix 30 hop、未来400 hop。
  prefixの2,160 energy値と未来316,800値を再計算した。
  最大絶対差は `3.642919299551295e-17`。

両方とも36 case、396実行可能分岐、72不適格、108 release／gap tail条件を検証した。
各取得469ファイルのSHA-256をverificationへ保存し、登録時に実ファイルと再照合した。
480版の未来396 PCMは旧v1素材と全byte一致し、旧形式へkey名を戻して新prefix項目を除いた
468分岐記録も一致した。これは旧条件の音響帰結を変えずに前史を追加した検査であり、
512版と480版の音声同一性は要求していない。

verifierにはprefix／futureの同じhop集計を使わせ、caseのbody／routing／活動／modulatorの
一致も検査した。3 Python検査で物理hop幅、routing、欠測・部分窓・非有限値を確認した。
取得済みファイルを変更しない読み取りoverlayで、誤hop、誤支持、誤prefix routing、
誤energy、1 sample欠落、非有限PCM、silent prefixへの活動混入、prefix比較失敗、
case identity違いの9条件を与え、すべて拒否した。

[物理grid素材記録](i10-action-profiles-physical-grid.json)にsource／binary／manifest／
verificationのhash、検査結果、再実行コマンドを保存した。
全Rustは1,090成功、失敗0、無視30（31 suite）。
`cargo test exit=0 @ 2026-09-18T02:55:05+09:00`。
通常Clippy、例のClippy、全target check、fmt、diff checkも通過した。

I10全体は未完了。次はこのprefixと未来PCMを同じNSGT／raw数値経路へ通して、
発行前の入力と反実仮想の実音targetを分離した特徴比較を作る。
8 medoidのrecipe・範囲に適合する行動profile、7 classの通常候補入力、全headの投影方針、
共有帰結表、actual-Voice/defaultのpaired差分、両busの全候補診断、4／16／64 Voiceの
全費用測定は引き続き未完。I9は保留し、新しい生成作用を有効にしない。

### 前史付き実音targetの特徴抽出（実装前登録、2026-09-18）

512-hopの登録済み396分岐を、私有音の通常 `body::Lane` 経路へ通す。
各bus・各分岐でsource単独のprefixから解析し、発行時にNSGTと特徴履歴をリセットしない。
取得条件は既存prototypeと同じ48 kHz、nfft 2,048、hop 512、右寄せcoherent、
Log2Space 55–8,000 Hz／96 bins per octave、RtConfig既定値、accent尺度0／1とする。
これはアプリ既定nfft 16,384の検査ではない。既存medoidへの行動template登録も含めない。

prefixは発行時までに利用可能な入力として保存し、未来は反実仮想の実音targetと明記する。
各hopの十生特徴、energy、原支持、NSGT warmup、公開された六身体記述値を保存する。
独立検査用に同じhopのenergy scanもlittle-endian f64で保存する。
候補ごとにprefixの全記録一致を検査し、Pythonで全energy・十生特徴・時刻支持を再計算する。
wait／skip／continue、release／gapの同条件音響一致、delayed action前の一致、
routingによる既知無音と両busの対応も確認する。入力の登録hashは取得前後に照合する。
取得器は明示実行する内部testとし、公開APIや楽器の音声保存能力を増やさない。


### 前史付き特徴targetの取得・独立照合（2026-09-18）

明示実行する内部test `body::action_targets::acquire_action_feature_targets` を追加し、
登録済み512-hop素材を通常の私有 `Lane::push` へ通した。各分岐・busは同じprefixから
独立に開始し、発行時にNSGT・raw・身体窓の履歴をリセットしない。
未来は `counterfactual_actual_audio_target`、前史は `issue_prefix` と記録した。
取得は13.87秒で終了した。これはoffline取得器の所要時間であり、RT負荷測定ではない。

`target/i10-action-features-512-20260918/` に36 case・396実行可能分岐を保存した。
各case・busのprefixは29 hop、各分岐の未来は375 hop。合計2,088固有prefix hopと
297,000未来hop、未来の六身体記述値公開30,096件を含む。
各caseの初回prefixと他の分岐のprefixを720回比較し、raw・energy・shape支持・身体公開記録が一致した。
同じ順序のLog2Space-aligned energy scanをf64 little-endianで保存した。

`collect_temporal_action_features.py` は登録済み入力469ファイルのhashを取得前後に照合する。
NumPyによる独立計算で、全hopのPCM energy、十生特徴の数値・欠測、時刻・原支持、
NSGT warmup、身体記述値の公開周期・個体世代を検査した。
未来だけで2,970,000 raw座標を比較し、最大絶対差は `5.417888360170764e-14`。
wait／skip／continueとrelease／gapの同条件288組が音響特徴・spectrum・energyで完全一致し、
遅延動作前の360組も完全一致した。非routingの264 streamは既知のゼロenergy・ゼロspectrumを保ち、
両bus routingの写しも一致した。取得した1,729ファイルのhashを検証記録へ保存し、登録時に再照合した。

初回のPython検査はrawの原支持で停止した。差分特徴は前hopのNSGT支持まで消費するが、
検査器が現在hopだけの `end - nfft` を期待していた。`max(0, end - nfft - hop)` へ訂正し、
Rust取得結果を変更せず全量再検査した。失敗した初回logと成功した再検査logを別々に保持する。
原支持を短く見せるためにRust側を変更していない。

保存結果への読み取りoverlayで、誤role、warmup再開、無音shapeの捏造、隣接energy支持の消失、
誤available時刻、誤energy、誤flux、身体世代違い、1行欠落、周波数grid違い、不適格候補の
target捏造の11条件を与え、すべて拒否した。元の取得ファイルは変更していない。
独立oracleの2 Python検査は、無音と欠測、warmup、差分、半octaveの帯域境界を確認する。

[特徴target登録](i10-action-feature-targets.json)へsource・test binary・manifest・verificationの
hash、件数、負例、再現コマンド、初回検査器の修正理由を記録した。
全Rustは1,090成功・失敗0・無視31（31 suite）。追加した取得testは通常では無視し、
今回は明示実行して1成功を確認した。`cargo test exit=0 @ 2026-09-18T03:10:33+09:00`。
通常Clippy、全target check、fmt、diff checkも通過した。

この検査はNSGT変換自体を別実装で検証したものではない。六身体記述値の数値も取得するが、
ここでの独立照合はその個体・支持・公開周期までであり、六値の数値検査は既存の別検査に属する。
私有source-onlyのenergy shareを、外部混合や共有groupのenergy shareへ転用しない。
8 medoidに適合する行動template、7 classの通常候補入力、全head投影、共有帰結表、
actual-Voice/defaultのpaired差分、両bus候補診断、4／16／64 Voiceの全I10費用測定は引き続き未完。
I9は保留を維持し、新しい生成作用も有効にしない。

### 観測・候補で共有する窓集計（実装前登録、2026-09-18）

候補headへ接続する前に、phraseの250 msのrise／decline／flux／energy shareの物理時間平均と、
2秒の低energy比率・平均RMS・log2-RMS端点傾斜を純粋な共通集計へ移す。
既存の90%支持条件、RMS floor、窓端を欠く傾斜の欠測、実hopをclipした中心時刻を維持する。
全標本にobserved／projected／unsupportedの既存タグを持たせ、観測と射影の物理支持率を別々に返す。
値が両方を消費する場合はprojectedとする。未来を観測支持へ加算せず、過去へ候補支持を逆流させない。

集計は順序付きの非重複区間だけを受け付け、窓外の行は消費しない。発行前のobservedと発行後の
projectedだけを使い、raw数値・energy・時刻・支持の不正は拒否する。窓は候補時刻で終えるが、
ordinal backoffに使う発行時coverageはこの移動窓の値で更新しない。head側の保持近似や
未知方針の全登録、7 class入力生成、共有帰結表は後続の必須工程として残す。
通常のphrase観測経路でも同じ式を使用し、保存済みtargetに対する独立窓集計と、
観測／候補の出所反転、欠測・端点・部分hop・90%境界・大きな絶対sample時計を検査する。


### 共通物理窓の通常接続と数値照合（2026-09-18）

`feature_projection::window` に純粋な窓集計を追加した。通常のphraseの250 msの四平均と
2秒のrecent-energy要約もこの式を呼ぶ。集計は呼出側の順序付きframeを一巡し、heap確保を行わない。
通常経路では二つの窓で使用する。追加・削減された計算の総費用は全I10測定へ算入する。

七出力の順序はrise、decline、flux、energy share、低energy比率、平均RMS、log2-RMS端点傾斜。
各座標にobserved／projectedの物理支持率を別々に持たせ、90%未満はunsupportedとする。
仮想入力を含む集計値はprojectedであり、観測済みとして公開しない。
低energy判定は既存のRMS floorと基準の1%を維持する。平均RMSはsqrt(energy)の時間平均であり、
窓energy平均の平方根に変更していない。傾斜には両窓端の実hop中心が必要で、端点欠測を
都合のよい内側のhopへ置換しない。

観測frameの終端・原音源依存・availableが発行時刻を越えれば拒否し、窓でclipしても
未来の観測値は使えるようにならない。projected frameは発行時刻以後だけで使用する。
既知無音、空窓、座標ごとの欠測、重複・逆順、不正値、支持90%境界、部分hop、
2^53を越えるsample時計を3 Rust検査で確認した。発行時のordinal backoff用coverageは
この関数へ可変状態として渡さず、移動窓の支持率で置換しない。

保存済み396分岐×両busを32時点、250 ms／2 s／8 sの3窓幅で数値replayした。
32時点は発行時から4秒までの均等点を最寄り整数sampleへ丸めたもの。
窓はbirth=0でclipする。未来の実音targetをprojectedタグへ置き換えたのはこの明示的な
数値fixture内だけであり、実音を予測値として採用するものではない。

`verify_temporal_feature_windows.py` のNumPy独立集計と76,032窓・532,224座標が一致した。
出所内訳はobserved 15,048、projected 515,592、unsupported 1,584。
最大絶対差は `2.1316282072803006e-14`。観測／仮想支持率は全座標で一致した。
入力targetの1,729ファイルは実行前後に登録hashと照合した。
結果は `target/i10-window-projection-20260918/` に保存した。

変更前のbinaryと変更後の通常rendererで、固定seed・3身体・混在routingの同じ素材を実行した。
WAVは全byte一致し、SHA-256は
`00ba126a393f45d4493fa56231229592d7010810ed7428c79b47c20853142a90`。
私有身体記述384件、自声帰結6件、将来身体記述予測6件も完全一致した。
共有reportは非同期の取得時刻に差があるため、同じbus・frameの56 snapshotを比較し、
phrase全状態と201群のrecent-energy要約が一致した。各runの14 unmatched cutは未比較として残す。
全report行の一致は主張しない。

feature manifestの四つのphrase平均へ候補射影方針を追加し、七集計の出所・支持・時計・
backoff不変条件を `i10_window_feature_projection` に記録した。
[窓射影登録](i10-window-feature-projection.json)へsource・binary・数値結果・通常replayのhashと
再現コマンドを保存した。全Rustは1,093成功・失敗0・無視32（31 suite）。
新しい数値replay testは明示実行で1成功。`cargo test exit=0 @ 2026-09-18T03:25:34+09:00`。
通常Clippy、全target check、fmt、diff checkも通過した。

I10全体は未完了。今回の集計は候補入力を受けられる数値経路であり、通常runtimeの
7 class候補producerではない。身体条件に適合する行動profile、全headの残る特徴・保持近似・
未知方針、共有帰結表、actual-Voice/defaultのpaired差分、両bus候補診断、全I10資源測定を続ける。
I9保留、新しい生成作用の無効、予測精度・可聴性・作者採用の別判定を維持する。

### 7 classの身体入力と有界時刻生成（実装前登録、2026-09-18）

offline取得器内のclass判定を共有する純粋な入力生成へ移す。出力はexcitation、release、
再考時刻、現在機会の消費、固定されたgap終端であり、音響energyやスペクトルではない。
入力には発行時刻・候補時刻・固定intrinsic period・現在の未消費機会・再考可能性・
候補時刻の身体活動とclass固有の身体許可を渡す。呼出側の身体許可が不明なら不適格とする。

onset-nowはゼロ遅延、delayed-onsetは正遅延、waitはpending opportunityと発行から50 ms以上の
再考時刻、skipはゼロ遅延の未消費due機会、continueはゼロ遅延の活動身体を必要とする。
releaseは候補時刻の活動を必要とし、gapは活動時のみreleaseして、活動の有無によらず
候補時刻＋発行時に固定したperiodまでexcitationを控える。gapのrelease tailは保持する。
この関数は方策、時計、実音、memoryを変更しない。現在の身体default選定そのものは後続課題。

時刻集合は12均等点、変更しないdefault、最大3追加時刻の計16以内とし、重複を除去する。
追加時刻は支持済みhorizon内だけを受け付ける。追加時刻を分位点として扱うには呼出側の
校正済みarrival根拠が別途必要で、今回その根拠は作らない。既存の私有trace診断は追加点なしで
同じ時刻生成を使い、元の13点以下・音声・学習を維持する。traceのevent時刻をaction時刻と混同しない。

共通APIを使う外部consumerは `examples/temporal_action_profiles.rs` とするため、
`life::action_candidates` のこの型・関数のみ公開する。取得器の396分岐について既存PCMと
class metadataが不変であることを照合し、法則の境界・不適格・固定終端・時刻容量を検査する。
通常の7 class音響producer、prototype transfer、共有帰結表、actual/default対応は引き続き必要。


### 共有した7 class身体入力則と時刻生成の検査（2026-09-18）

`life::action_candidates` を追加した。`Class::input` はclass、発行・候補時刻、固定period、
候補時刻の活動、身体許可、pending／due状態から、excitation／release／再考の時刻、
機会消費、gap終端を返す。身体許可は `Option<bool>` とし、未知を許可へ変換しない。
活動不明ではcontinue／release／gapを構成しない。gapだけが既知の正periodを必要とする。
period不明を理由に、periodを消費しないrelease等まで不適格にはしない。

onset-nowとdelayed-onsetのゼロ／正遅延、waitのceil(fs/20)境界、skipのpendingかつdue／未消費、
continueのゼロ遅延、候補時刻に既に失活したrelease、活動・非活動のgap、固定period変更後も
元gap終端が変わらない条件、非有限ではなく整数終端のoverflowを3 Rust検査で確認した。
この入力はwaveform・energy・特徴支持を持たず、音響モデルの支持を代わりに宣言しない。
Noneは入力を構成できない意味であり、実音がゼロという意味ではない。

同じmoduleの時刻生成は12均等点＋default＋最大3追加点を、16要素の固定配列へ作る。
重複・支持範囲外の追加点・整数overflowを除去し、defaultはspan外でも保持する。
小さいspanで丸めが重複する場合、spanなし、ゼロspan、2^53を越える絶対時計も検査した。
normalの私有trace診断はこの共通算術へ置換し、既存のevent時刻・最大13点を維持した。
追加点は渡さず、校正済み分位点の生成を実装した扱いにはしない。
既存のtrace検査で二回のpreview＝26 query、readonlyの分布、50 ms制限、世代退役を再確認した。

外部consumerの `examples/temporal_action_profiles.rs` はclass判定と指令時刻をこの共有入力から作る。
この利用のためにClass／BodyState／Inputを公開したが、時刻生成関数はcrate内に留めた。
取得器は明示された登録身体の状態だけを渡し、ライブのecology許可や身体状態を推定していない。
新取得先は `target/i10-shared-action-inputs-512-20260918/`。

独立verifierは36 case・396実行可能分岐・72不適格・108 tail条件を再検証した。
元512-hop素材と、新取得の469ファイルがhashまで一致した。内訳は396未来PCM、36前史PCM、
36 profile JSON、manifest。独立検証JSONも一致した。
前史2,088・未来297,000 energy値の最大絶対差は従来と同じ `3.2959746043559335e-17`。
wait／skip／continueの音響一致と機会操作の違い、release／gapでtailを消さない性質を保持した。

feature manifestへ `i10_action_input_contract` を追加した。
[身体入力登録](i10-action-inputs.json)にsource・binary・verificationのhash、元素材との比較、
再現コマンド、未実装範囲を保存した。全Rustは1,096成功・失敗0・無視32（31 suite）。
`cargo test exit=0 @ 2026-09-18T03:40:31+09:00`。
通常Clippy、例のClippy、全target check、fmt、diff checkも通過した。

I10全体は未完了。この変更は身体入力と時刻候補の共通核までであり、通常の7 class音響producer
ではない。ライブ身体・世代・方策からの事実取得、body-defaultのclass対応、適合する行動profile、
全head射影、共有帰結表、actual/default差分、両bus診断、全I10資源測定が残る。
I9保留と新しい生成作用の無効を維持する。

### 実Voiceの身体default事実取得（実装前登録、2026-09-18）

通常Voiceがphonation engineへ渡す生存・onset emission gateを時刻付きで保存する。
これは将来候補の完全な身体許可ではなく、そのhopの入力事実である。rendererは指令適用前に
source/body世代、活動中と予約済みのTone、両busのroutingを取得する。活動はenvelopeの
開始／終了による定義であり、可聴性やエネルギー支持とは区別する。

診断defaultは既存順序で最初の受理済みonset/releaseを優先し、それがなければ活動身体の
continue、予約済みonsetの順とする。受理されたreleaseでもonset前の取り消しはrelease classの
活動条件を満たさない。未予約の次回機会や再考時刻は時計・乱数を進めずには確定できないため、
未対応を明示する。この段階でidle時のdefault対応全体を完了とはしない。

最大64所有者、各20 Hz以下のreadonly採取と最大64出力を事前確保し、世代交代・退役・
容量超過・重複batch・過去時計を検査する。recordは新しい方策選択や候補の実行ではない。
reportなしでも毎hop出力をdrainする。実rendererの受理／拒否と、既存音声・既存予測の
不変性を検査する。I10全体の音響producer・共有帰結表・全資源検査は引き続き必要である。

### 身体defaultの通常接続と独立照合（2026-09-18）

`PhonationBatch::body_policy` に実Voiceがengineへ渡した生存とonset emission gateを
hop時刻付きで保存し、collectorは無発音Voiceのbatchも保持する契約に変更した。
batchの存在は発音を意味しない。公開APIを使う発音検査は実onsetの存在を確認するよう変更し、
生成音声の検査は維持した。内部の再利用bufferも全Voice分を保持する。新しい診断Bankと
出力Vecは初期化時に各64枠を確保し、hop内では拡張しない。全Voice batchの初回／増員時の
確保は従来のcollectorの確保経路を使い、既定数を無制限な音響モデル容量とは扱わない。

rendererは指令適用前のToneをsource IDのBTreeMap範囲から読み、同一source世代の活動数、
両busの活動数、予約数と次の予約onsetを記録する。旧source世代と不一致／未知body世代のToneは
別の件数で保持する。身体の活動は `onset <= now < release_end` であり、実測音圧とは異なる。
最初の受理済みOn/Offを既存順にclass化し、拒否指令では置換しない。最初の受理指令がclassへ
対応しなければ、後続の対応しやすい指令へ飛ばさず `accepted_transition_unmapped` とする。
複数の受理指令数と最初のTone ID、実routing、body世代も保存する。

指令がなければ活動身体のcontinue、予約済みonsetの順でdefaultを記録する。
未予約の未来機会は `no_planned_transition` とし、waitや架空の無音へ変換しない。
policyは元hopの事実であり、将来のrate・pitch・確率的許可の代用にはならない。
このmappingはversion 1、各source世代20 Hz以下のreadonly診断である。採取間の指令は
既存self-sound outcomeへ残るが、今回のdefault recordにはすべてを収録しない。
通常reportに `body_default`、snapshot/UIに対応数・未知数・dropと最新defaultを追加した。
reportなしでも出力をdrainする。新しい候補の評価や実行は行わない。

4件のRust検査を追加し、世代違い／過去時刻の入力、未知policy、非有限period、最初の未対応指令、
ceil(fs/20)の採取間隔、64枠と出力上限、退役・世代交代、重複batchを確認した。
実renderer検査では予約onset→活動→release、拒否指令との順序、複数受理、元Toneのrouting、
旧source世代のtail、onset前キャンセルと過去hopの非採取を検査した。同時に両busの音声を
診断なしrendererと完全一致させた。実Voiceのcollector検査も無発音hopのpolicy保持を確認した。

通常binaryの素材と結果は `target/i10-live-default-20260918/` に保存した。
三身体・両bus素材は152 record（onset-now 3、continue 149）、無音Voice＋周期Voice素材は
116 record（release 3、continue 17、未対応96）を取得した。後者にはgate閉鎖58 recordを含む。
`verify.py` は既存の受理済みcommand記録と発行済みenvelopeから、採取時点の活動・予約・
routing・世代対応・最初の受理指令・class／時刻・採取間隔・summary件数を独立に再構成した。
268 recordが一致し、容量／出力dropは0。正常素材のbody世代不一致が0であることは、
一般のprototype transferや未知身体の適合を証明しない。

三身体素材は変更前と変更後のobserve/off×report有無4条件でWAV hashが一致した。
SHA256は `00ba126a393f45d4493fa56231229592d7010810ed7428c79b47c20853142a90`。
周期素材も新旧WAVが一致し、SHA256は
`6346c539116b3aceb413b1ad6d4f5ff8aa5c4a04434b56f675876500ff15dcf3`。
両素材合計で既存body descriptor 498件、self-sound outcome 14件、descriptor prediction 14件、
onset 7件、population step 1,321件が新旧完全一致した。全headや非同期公開時刻の一致を
この照合の対象には含めない。初回の旧batch契約によるテスト失敗と、素材のcycles型誤りの
ログも保存し、修正後の結果と区別した。

全Rustは1,100成功・失敗0・無視32（31 suite）。
`cargo test exit=0 @ 2026-09-18T04:07:16+09:00`。
通常Clippy、全target check、fmt、diff checkも通過した。検査ログ、独立照合JSON、入力、
新旧binaryとsourceのhashを同artifactへ保存した。feature manifestの身体入力契約へ
この通常consumerと未対応範囲を追記した。

I10全体は未完了。残るdefault対応には未予約のintrinsic／再考時刻と音響入力の対応がある。
候補時刻の七class適格性、身体条件に適合する行動profile、全head射影、共有帰結表、
actual/defaultのpaired差分、両bus候補診断、全I10資源測定を引き続き実装・検証する。
I9保留、新しい生成作用の無効、R/Aの校正・可聴性・作者採用の別判定を維持する。

### 登録medoidと同じ前史からの行動素材取得（実装前登録、2026-09-18）

既存8-medoidモデルのrecipeと先の396分岐のrecipeは異なるため、後者を既存prototypeの
行動templateとは扱わない。モデルを上書きせず、各medoidの元scenarioを通常workerで再生し、
そのdescriptorの実end sampleでrendererの私有Tone状態を凍結する。requested cutは実endと
区別する。凍結前の実Voice指令を別rendererへも与え、両busの私有前史を単一Voiceの通常pre-guard bus出力と照合する。

取得用probeは明示実行するignored test内だけに設け、通常buildには含めない。
各medoidで、発行から4秒までの32整数時刻、7 classの定義に沿った分岐を実音で比較する。
新onsetのrecipeは発行時の実Voiceの身体・振幅・周波数・modulatorと元のenvelopeから固定し、
全候補へ共用する。既存Toneは実rendererのstateからforkし、前史を指令から再生したrendererと
各分岐の全PCMを比較する。通常workerで未発行の将来指令は借用しない。

素材のdue／pending／許可と0.2秒periodは明示した反実仮想条件であり、元のsustain方策の
intrinsic periodを観測したものではない。hold方策の未知periodや未予約機会をこの素材で埋めない。
発行後のrhythmは発行時の値から固定周波数で進める条件とする。実際の将来rhythm予測ではない。
新しい音響特徴・共有帰結表への入力には、この条件範囲とmodel/versionの対応を必要とする。
今回の取得だけでprototype転移精度や校正、通常の候補producerを完成とはしない。

### 同一medoid前史の行動実音・生特徴（2026-09-18）

`runtime/body_profiles.rs` の明示実行するignored testを追加した。通常workerの単一Voiceから、
元medoidのdescriptor実endでTone状態をforkする。requested cut 9,600／38,400／96,000に対し、
実endは5,120／35,840／92,160であり、こちらを発行時刻に使う。別rendererへ同じ発行済み
指令を渡し、通常workerの両bus PCMと前史全体を完全一致させた。8 medoidの元descriptorも
raw値・mask・支持・coverageまで一致した。元モデル
`b3eb8cba69007e16aedb013d9ee8ba617b3ecb4d3420afb492768bfeba02fd28` は変更していない。

新onsetには発行時に利用可能な最新Voice control（直前hopで評価）、身体・周波数・振幅・
modulator、元envelopeを固定して用いる。sustain身体の現在Toneを保持したまま新しいToneを
加える。releaseは既存Toneを終了させ、gapは同じtailを保つ。32時刻のうち各classの時間条件を
満たす129分岐を8 medoidで取得し、計1,032分岐を実Tone forkと独立指令再生で全PCM照合した。
4秒境界の動作には、この4秒素材内の動作後サンプルがない。素材の未来を観測支持と呼ばず、
現方策の未予約機会や未知periodを補う根拠にも使わない。probeは `cfg(test)` のみであり、
instrument／通常renderer binaryには音声保存経路や追加のper-hop処理を導入しない。

取得過程でJSONのf64往復不一致を検出した。元medoidのflux
`0.012374198076345993` が読込・再保存で `0.012374198076345991` になっていた。
`serde_json` の `float_roundtrip` を有効にし、実medoidの6値と符号付きゼロのbit同一性を
通常Rust回帰検査へ追加した。許容差で吸収せず、失敗素材を残した上で後継runへ取得した。
coverageの別の不一致は、f32の直接JSON表現とValueへのf64昇格の違いだったため、
宣言型f32へ戻したbit一致で照合した。raw f64値の比較は緩和していない。

`verify_temporal_medoid_actions.py` は元モデルのcanonical hash、scenario／report／WAV／config
hash、8件のdescriptor、classと時刻・機会操作、両busのrouting、PCM支持、774,000個の
物理hop energy値を独立に照合する。最大絶対energy差は `1.0408340855860843e-16`。
同値classの1,024組と動作前PCMの1,488組が一致した。release tailの465条件中、
323条件は1e-6を越え、142条件は自然減衰によって閾値以下だった。全tailを可聴と要求する
検証器の誤った仮定を修正し、既知の小さい実音を欠測やゼロへ置換しないことを維持した。
case欠落・発行時計・gap終端・元descriptor・routing・energyの6種類の読取overlay改変も拒否した。

通常私有Laneを使う特徴取得器を、このmedoid素材の可変長前史にも対応させた。
2 bus・1,640前史hop・774,000未来hop、7,740,000生特徴座標と76,368未来body publicationを
取得した。前史を分岐ごとに再処理し、2,048回の前史一致を確認した。各medoid自身の前史末端
descriptorが元登録に戻ることも検査した。NumPy独立計算の最大生特徴差は
`5.861977570020827e-14`。支持mask・物理時計・NSGT warmup・PCM energy・routing・同値class・
動作前特徴が一致した。NSGT変換と未来の六身体窓記述値の数値式は、今回独立再実装していない。
元36 case・396分岐の旧特徴素材も同じ更新後verifierで再照合し、件数と誤差が不変だった。

最終取得は検証済みtest executableを固定し、PCMを `target/i10-medoid-actions-frozen-20260918/`、
特徴を `target/i10-medoid-features-frozen-20260918/` に保存した。元モデルと素材のhash、取得・
検証のコマンド、source／executable hash、各出力hashの一覧は前者の `registration.json` と
両者の `verification.json` に保持する。feature manifestへ対応と未対応範囲を追記した。
初回・途中の失敗素材とログは削除していない。

通常binaryでも以前の三身体素材を再生し、WAV SHA256
`00ba126a393f45d4493fa56231229592d7010810ed7428c79b47c20853142a90` が維持された。
身体default 152件、body descriptor 384件、self-sound outcome 6件、descriptor prediction 6件、
onset 3件、population step 750件も変更前と一致した。
全Rustは1,101成功・失敗0・無視33（31 suite）。
`cargo test exit=0 @ 2026-09-18T04:33:36+09:00`。明示取得2 test、Python 5 test、
通常Clippy・全target check・fmt・diff checkも通過した。

この素材は同じmedoid前史を持つ条件付き実音targetであり、通常の音響予測モデルではない。
profileの支持範囲・補間／転移とactual-Voiceとの差、全head射影、共有帰結表、未予約時刻を
含むdefault対応、paired差分・両bus候補診断、全I10資源検査が残る。I10全体は未完了。
I9保留、新しい生成作用の無効、校正・可聴性・作者採用の別判定を維持する。

### 条件付きprofile読込と候補特徴の接続（実装前登録、2026-09-18）

検証済みmedoid実音特徴から、f64値とmaskを保った物理hop profileを作る。同じ音響軌道だけを
byte一致で共用し、classごとの機会操作や動作時刻は共用しない。prototypeは最大8、32動作時刻、
7 class、4秒・375 hopを上限とする。入力ファイルのSHA-256、身体model version、medoid順序、
sample rate／FFT／hop、長さ・mask・有限値を起動時に検査し、両busのworkerで同じimmutable
profileを共用する。offでは読込・保持しない。設定は研究用であり、新たな生成作用は有効にしない。

profile値は観測した未来ではなく、固定身体からの未校正なdescriptor transferとして扱う。
候補時刻と帰結評価時刻をAPI上で分離する。時刻定義の判断が未確定な間、共有帰結表の
評価時刻の仕様変更は行わない。観測prefixは元の支持・availabilityを保存し、未来profileは
Projected、未対応座標はUnsupportedとする。観測へのcommitや学習は行わない。

最初の未来hopのrise／declineは、発行時に利用可能な隣接観測log RMSから再計算する。
その境界のfluxは実prefixのスペクトルがない場合は不明とし、prototype前史のfluxを借用しない。
未来のenergy shareは、候補group energyと発行時に固定した背景energyから計算する。
背景をown-excluded予測と呼ばない。pitch／gain／rateの未登録変換・動作時刻の補間は行わない。

### 条件付きprofileと通常候補特徴窓の接続結果（2026-09-18）

**通常workerでのprimitive特徴窓まで接続した。全headの帰結表とI10全体は未完了。**
`scripts/compile_temporal_action_profiles.py` で、既存8 medoid・1,032分岐の検証済み特徴を
固定profileへ変換した。音響軌道のbyte一致だけを共用し、442軌道・165,750物理hop、
15,934,505 byteになった。`target/i10-action-profiles-20260918.bin` のSHA-256は
`033905eadd9a7c681a8568ea347c25d1c1482d49d708bc849b00607a85d51fb5`。
別の読戻し検証器で全bindingの387,000 hop・4,257,000座標を元JSONLとbit照合した。
欠測601,255座標と既知ゼロ1,124,016座標を区別したまま一致した。

`[temporal_action_profiles]` の `file` と `sha256` を明示する研究用設定を追加した。
`temporal_body_prototypes` と `temporal_phrase` を前提とする。起動時に容量・hash・物理格子・
モデル識別子・medoid順序／mask／六値のbit・classと時刻のbinding・有限値を検査する。
通常observeでは一度だけ読み込み、両busでimmutable `Arc` を共用する。
offでは読み込まず、候補表も確保しない。実デバイスと登録sample rateの不一致も拒否する。

通常の共有group割当から、最大8×7×32 cellの特徴窓を各bus workerで作る。
割当の更新時点に合わせて再計算し、group退役時は更新周期を待たず除外する。
直近2秒またはgroupの生存期間の実前史へProjected hopを連結し、既存の物理窓集計を使う。
前史のsource／availabilityを保持し、発行時coverageを未来の特徴から増やさない。
背景energyは発行時の観測energyとshareから推定して固定する。ownとshareがともにゼロなら
背景は不明とする。各cellにはこの固定背景も残す。未登録の動作時刻を補間しない。
reportにはhash・identity・時刻・容量・cell数・支持座標数と各classの最後のcellを保存し、
UIには未校正の動作時刻窓であることを明示する。

APIでは動作時刻と評価時刻を分離した。通常表の評価時刻は、既存仕様どおり動作時刻のまま。
したがって、その時刻に始まるonset／releaseの後続音響効果を評価したとは扱わない。
動作後まで評価する仕様への変更は未確定であり、共有帰結headへ暗黙に採用していない。
これはprimitive窓の表であり、section／phrase等の全head特徴やordinal帰結列ではない。

Rust回帰検査では、破損header／payload・誤モデル・binding・mask、既知ゼロ、境界flux、
実前史の利用可能時刻、混合窓の出所と数値、独立した動作／評価時刻、発行coverageの固定、
group退役と観測snapshot不変を検査した。通常rendererでは、両busの18 snapshotで非空の
候補表を確認した。完全に未来に入る72窓・504座標を保存profileから別計算で照合した。
同一frameの67観測（両busの終端を含む）も配送時間と追加診断以外は一致した。
reportの間引き時刻は発行時刻と一致しなかったため、実前史の全hopと発行coverageの直接照合を
このreport比較の実績には含めない。そこはRustの混合窓・固定coverage検査による。

既存onset 3件、身体default 152件、body descriptor 384件、自声outcome／descriptor予測各6件、
population step 750件が不変だった。profile有無・report有無・offでWAVも同じ
`00ba126a393f45d4493fa56231229592d7010810ed7428c79b47c20853142a90` を保った。
欠落fileはobserveでは起動エラー、offでは成功した。
全Rustは1,105成功・失敗0・無視33（31 suite）。
`cargo test exit=0 @ 2026-09-18T05:10:13+09:00`。
Clippy・全target check・fmt・diff check・Python構文検査も通過した。

実行コマンド・設定・report・検証器・検証結果・source／binary hashは
`target/i10-compiled-profile-live-20260918/`、登録は
[条件付きprofile接続記録](i10-conditional-action-profiles.json) に保存した。
残りは評価時刻の確定、全head射影と未知質量を含む帰結混合、actual-Voice/default対応と
両busのpaired差分、4／16／64 Voiceの全I10資源検査。I9保留とI11生成作用の無効を維持する。

### 候補articulationとphrase入力の接続結果（2026-09-18）

**条件付き状態予測とphrase入力を通常候補表へ追加した。全headの帰結列は未完了。**
`gesture/projection.rs` は、発行時に保持していたarticulation pathの状態・入場時刻・質量から
scratch予測を行う。観測側と11座標の標準化／欠測mask、softplus遷移率、低energy時だけの
gap遷移則を共用する。未来には候補のrise／decline／fluxと実時間の経過を使い、ridge motionは
未対応としてmaskする。予測内で観測の忘却・未知親からの新規採用は行わず、元の未知質量を
未来音で回収しない。既存観測側の採用・忘却則は変更していない。

保持15 path・展開60 pathの固定配列を使い、同じ状態／入場時刻を合算してから、質量と
安定した状態／時刻順で枝刈りする。落とした質量は未知へ加算し、生存pathへ再配分しない。
予測呼出し内にheap allocationはない。連続したProjected hop、group／epoch、発行時計、
4秒上限を検査する。最終hopは評価時刻までのdtで遷移させるが、入力特徴値は登録済みの
512-sample hop値を使う。端数PCMを再解析する方式ではない。合成した観測Runやevidence
endpointを作らず、元のGesture／Phrase／memoryを更新しない。

候補cellへ4状態の質量、発行時の未知、枝刈り質量、合計未知、時刻と出所flagを追加した。
発行時刻そのものの値は実状態のまま、未来時刻の値は条件付き予測とする。
phraseのv5–v8は、共通窓集計を250 ms窓へ適用したrise／decline／flux／energy share。
v9の2秒groupingは観測側と同じ関数を使い、利用可能な実前史だけで90%支持を満たす場合に
計算する。未知の未来accentやwordを生成して補わない。v10のordered matchは発行時刻の
対応値だけを使い、未来のordered比較は未接続。v11／v12は現行base観測headと同様に欠測を
維持する。v1–v4は各articulation状態のone-hotとして後続で条件付き評価し、状態質量の平均を
非線形headの入力へ代入しない。これらの投影方針をfeature manifestへ登録した。

Rustでは1／256／512 sampleの閉形式遷移、gap gate、未知100%からの非採用、欠測motionの
係数とelapsed尺度、入力欠落・誤identity・非因果時刻、4秒の枝刈り質量保存、元状態不変を
検査した。phrase側では候補窓とarticulationの時刻整合、2秒groupingの90%境界とavailabilityを
検査した。正常な観測経路の計算順序を保ち、通常rendererでWAVと既存診断の不変性も確認した。

通常reportの56 cell記録をidentityで重複排除すると28 cellで、未来16・発行時刻12だった。
重複28件は値も完全一致した。未来16 cellの状態質量・未知・時計と、250 ms窓の64座標を
検証した。250 ms数値は保存profileから別計算した。発行時刻のarticulation snapshotと直接
同期したreport比較は0件であり、この同一性はRust検査の結果として区別する。
別途、従来の2秒窓の140座標も再照合した。同一frameの35観測と両bus終端は、追加診断と
配送時間を除いて変更前と一致した。全report frameの同時刻一致を確認したという意味ではない。
候補計算による追加費用と非同期reportの間引きを認め、全I10の実時間合格とは扱わない。

onset 3件、身体default 152件、body descriptor 384件、自声outcome／descriptor予測各6件、
population step 750件が変更前と一致した。変更前・変更後・reportなし・offのWAVは同じ
`00ba126a393f45d4493fa56231229592d7010810ed7428c79b47c20853142a90`。
全Rustは1,109成功・失敗0・無視33（31 suite）、
`cargo test exit=0 @ 2026-09-18T05:30:49+09:00`。
Clippy・全target check・fmt・diff checkも通過した。

設定・実行記録・report・検証器・結果・hashは
`target/i10-articulation-profile-live-20260918/`、登録は
[候補articulation接続記録](i10-articulation-projection.json) に保存した。
次は発行時に保持したexpected segmentとの比較をclosure／continuation入力へ接続する。
全headのalternativesとscore、固定issue coverage・group weightを使う混合、defaultとのpaired差分、
評価時刻と未予約default、全資源検査は引き続きI10の必須残項目。I9は再開していない。


## 2026-09-18: 固定expected segmentと候補のraw ordinal head

発行時に保持したexpected segmentを候補の十生特徴と比較し、closureの残差入力と
経路別continuationを通常workerの候補cellへ接続した。登録は
[expected segmentとraw headの検証記録](i10-expected-segment-projection.json)。
全headのalternatives、重み更新、backoff、校正済み評定や生成列の完成を意味しない。

過去部分は、観測時に保存した残差と実際の比較区間をそのまま使う。未来部分だけを、
発行時に利用可能な一つの参照と比較する。参照のgroup／epoch／bus、query／episode、
原source／availability時刻を保持し、expected区間の外へ延長しない。共通座標の標準化
二乗距離を平均し、2秒またはgroup年齢の窓内で比較支持時間による平均を取る。
未知をゼロ距離に置換せず、観測・予測それぞれの平均と支持sample数をreportへ保存する。
過去を新しい参照で再採点せず、memory、比較回数、私有traceも更新しない。

closureはこの残差と既存の欠測maskを既存raw ordinal式へ渡す。continuationは残差に加え、
発行時刻だけの到来確率、各phrase pathの開始時刻から計算した候補評価時の生存率、
実観測支持付きaccent密度とgroupingを使う。未来の到来確率やaccentを捏造しない。
非線形ordinal式は保持pathごとに評価し、元のpath質量と未知質量を別々に保存する。
生存率は既存base headと同じarticulation周辺化であり、未知状態を既存のmissing stateとして
扱う。このraw診断を、全帰結表の解決済み質量や校正済み不確実性として流用してはならない。

Rustでは元の比較区間、参照の不一致・未来availability・失効、ゼロ距離と欠測、
accent密度の90%境界・保持上限による欠測・利用時刻、発行時の通常continuationとの一致、
未来のarrival欠測とpath重み不変を検査した。通常の観測結果も変更前binaryと比較した。

実音検査は持続音と段階的振幅変更の2素材、両bus。従来の8-hop memory spanでは
発行時の未来expected segmentが得られなかったため、96／160-hopの別条件を明示した。
閉形式で生存率を検査できる未適合の診断係数（hazard = log(2 + 経過秒)、closure残差係数
-0.25、continuation生存率係数2）を最終比較render前に固定した。採用済み設定ではない。

保存profileから独立に再計算し、1,576 cellのうち未来参照比較22件・9,412 sample、
closure分布376件、continuation分布23,640件を照合した。生存率の最大積分誤差は
1.465e-8で、既存の積分許容値 `1e-9 + 1e-7 * integral` 内。
過去残差の平均そのものは間引かれたreportから再構成していない。ここはRustの因果的窓検査と
通常観測の不変性に分け、独立検証の範囲を拡大解釈しない。

変更前後で同一frameの観測236件と両素材・両busの終端が一致した。
onset 2件、身体default 253件、body descriptor 252件、自声outcome／descriptor予測各4件、
population step 1,259件も一致。各素材の変更前・変更後・reportなし・offのWAVは同一。
全Rustは1,114成功・失敗0・無視33（31 suite）、
`cargo test exit=0 @ 2026-09-18T05:59:12+09:00`。
Clippy・全target check・fmt・diff check通過。設定・report・実行記録と結果は
`target/i10-residual-profile-live-20260918/`、独立照合は
`scripts/verify_temporal_projected_heads.py` に保存した。

I10全体は未完了。残るhead入力・alternativesとscore、固定したissue coverage／group weightに
よる混合とbackoff・未知質量、candidate/defaultのpaired差分、actual-Voiceと未予約時刻、
評価時刻の決定、4／16／64 Voice・両busの全資源検査を引き続き完了条件とする。
I9保留とI11作用無効を維持する。


## 2026-09-18: 発行時の音群重みを固定したraw head混合

現行base phraseの候補headを、発行時の観測coverage・音群重み・priorで混合する通常経路を
接続した。登録は [raw head混合記録](i10-raw-head-mixture.json)。以前から独立数値検査していた
`ratings::project_rating` を通常候補診断で使い、まだ使わない他の数値核はtest-onlyのまま残した。

発行時の通常2秒窓またはepoch年齢のcoverageと各音群の重み、known mass、headのpriorを
一度保存する。選択音群だけを候補の各phrase pathのraw headに置換し、それ以外の音群の
カテゴリ分布と重みは変更しない。保持pathの発行時質量を用い、欠測headの質量を既知へ
再配分しない。支持付きpath／group分布を混合した後、現行base headのtemperature=1を適用し、
`r=e_issue*s` と固定priorでbackoffする。未来窓のうち既に聞いた比率をe_issueへ代入しない。

reportの `raw_unreweighted_heads` はclosure／continuationのカテゴリ分布、期待評定、
固定coverage、選択音群・他音群それぞれの支持質量と合計、coverage適用後の支持を持つ。
表には発行時の全音群の重みと元の分布も保存し、独立検証できるようにした。
UIはrawかつ未reweightと明示する。最大7 held group＋15 selected pathのstack scratchを使い、
cellごとのheap allocationは加えていない。増えたcell／snapshotの大きさは既存table byte診断に含む。

この段階では共同scoreの更新を実装したことにしない。通常観測モデルは初期診断版であり、
§9.1が要求する共有contextと各音群local pathの共同beam・条件付きlog-potentialをまだ持たない。
既存のphrase質量や音群別section候補を、その共同事後分布と呼び替えることはできない。
今回のraw混合は後続の比較対象および固定coverage／probe集計の接続であり、
`q(C)*q(local|C)*exp(delta_log_score)`、帰結の解釈entropy、全headの完成ではない。
この不足を、仮のcontext重み1や未実装score差0で埋めない。

Rustでは発行時の通常ordinalとの一致、未来窓でも固定coverageが残ること、非対称な2音群・
2pathのカテゴリ混合、部分欠測の質量保存、全量prior、時刻不一致・質量不整合の拒否を検査した。
実音は前段と同じ持続音・振幅変更素材に、三身体・異なるbus routingの素材を加えた。
両busで1,618 cell・3,236分布を確率空間の別計算と照合した。
複数音群がともに支持を持つ混合28件、全量prior 1,200件、全窓が未来のcell 912件を含む。
発行時の通常snapshotと直接同期して比較できたheadは12件。残りを同期比較済みとは扱わない。

変更前後の同一frameの通常観測267件と、3素材それぞれの両bus終端が一致した。
onset 5件、身体default 405件、body descriptor 636件、自声outcome／descriptor予測各10件、
population step 2,009件と、全素材のWAVも一致した。
全Rustは1,115成功・失敗0・無視33（31 suite）、
`cargo test exit=0 @ 2026-09-18T06:12:43+09:00`。
Clippy・全target check・fmt・diff check通過。実行記録と結果は
`target/i10-raw-mixture-live-20260918/`、数値照合は
`scripts/verify_temporal_projected_heads.py` の `raw_mixtures` に保存した。

I10全体は未完了。共同alternativesと条件付きscore、残るtiming／invitation入力、
元の未知・枝刈り質量とhead支持の区別、actual-Voice/default対応とpaired差分、
評価時刻、全資源検査を残す。I9保留とI11作用無効を維持する。


## 2026-09-18: 共同正規化から候補帰結への数値経路

[共同正規化記録](i10-joint-normalization.json) の範囲で、§9.1の数値核を追加した。
全列挙された共有parent／extensionと局所parent／extensionを使い、各音群の分配関数を
共有重みに反映してから、共有7 resolved＋unknown、局所15 resolved＋unknownへ枝刈りする。
観測後にparentごとに再正規化する方式ではない。explicit unknownと枝刈り質量を別保存し、
保持したresolved pathへ再配分しない。大きな共通log-potentialを引いてからpriorを加え、
元の小さな重み差を消さない。欠測区間では全potentialを抑制し、遷移だけを使う。
不正な親prior・遷移正規化・重複slot・非有限入力を拒否し、直前の数値posteriorは保つ。

その保持結果を候補帰結核へ渡す経路も追加した。実際に保持した共有・局所pairの積だけを使い、
対応しない候補や重複を拒否する。候補の未支持質量と、元からのunknownを別のfieldで報告する。
候補帰結核のcoverage入力はhead別の3値に修正した。2秒／8秒の発行時窓を単一のcoverageで
代用せず、一つのheadがcoverageゼロでも、他のheadの支持と分布は維持する。

独立したDecimal・60桁の確率空間全列挙25ケースと一致し、fixtureの再生成も完全一致した。
最大32共有pair・8音群・各256局所pair、合計65,536局所pairの入力で、質量保存と安定tieを検査。
別の後続cueの反例では、枝刈り前の情報を全保持したcontext質量0.538953に対し、
捨てたidentityをunknownへ集約した経路は0.45となった。直前の分配関数が一致していても、
後の解釈は同じにならない。この差0.088953を、認知的忘却や採用済みの誤差として扱わない。
新規Rust検査7件、既存の候補帰結Decimal検査も通過した。

全Rustは1,122成功・失敗0・無視33（31 suite）、
`cargo test exit=0 @ 2026-09-18T06:31:06+09:00`。
Clippy・全target check・fmt・diff check通過。ログは
`target/i10-joint-normalization-20260918/`。
共同正規化と候補帰結はまだtest-onlyであり、今回の数値入力を通常観測に注入していない。

通常接続には、互換性のある成分tupleと共有proposal、pathの寿命、実際の条件付き特徴とscoreが
必要。現行manifestには完全なjoint observationのordered layoutがなく、stage 3の適合前に
登録する必要がある。観測ID・bus／epoch・一観測一更新の所有は、その実入力側で接続する。
head内の音群別部分支持、残るtiming／invitation、actual-Voice/default対応、評価時刻、
全資源検査も未完了。今回の数値核だけでI10完了とはしない。I9は引き続き保留。


## 2026-09-18: 局所成分の遷移と有界tuple合成

§9.1の局所proposal構築を `joint/proposals.rs` に追加した。登録は
`feature-manifest.json` の `i10_joint_proposals`。articulation、grouping、phrase、
section、correspondenceの順で、既知stayと明示的unknownを予約する。
採用した既知raw scoreを正規化してから、10／10／30／120／10秒の物理時間による
unknown遷移を一度適用する。既知のungrouped／no-memoryはunknownとは別IDで扱う。
phrase／sectionのstay／exit raw scoreと実際のidentityは入力側が供給する契約を保つ。

articulationは既存の観測経路と同じcompeting-rate関数を共有する。
unknown親からの既知状態への移行は、新しい観測支持がある場合だけ一様raw scoreで提案する。
新しいgapには観測済み低energyの条件が必要で、欠測から新しいgapを作らない。
既存gapのstayは欠測中にも遷移priorとして残る。新しいtrajectory状態は追加していない。

成分リストの積からall-unknown、合法なfull stay、phrase／sectionのstay-or-exitの
各組合せで最良の合法tupleを先に予約する。残りは隣接indexのbest-first heapで探索し、
最大16 pop、各pop最大5近傍、合計最大16 tupleに制限する。最後のpop後は展開しない。
遷移は採用tuple上で再正規化し、候補に入らなかった組合せをposteriorの枝刈り質量へ
読み替えない。合法性は、整列後のphrase／section entryの組合せ行列で受け取る。
それ以外の成分の適格性・実pathの互換性は、今後接続する実入力側の責務である。

重複priorityの検出には86,640 byteの固定indexを一度確保し、前回触れた最大86 keyだけを
クリアする。heapとindexは繰り返し再利用する。各tupleを既存cache全体と線形比較する
処理を除いた。これは局所scratchの構成であり、両bus・全Voiceの実時間合格ではない。

独立したDecimal・60桁の全Cartesian列挙12ケースと、選択順序・遷移確率・予約・除外を
照合した。最大ケースは54,150組合せを全列挙している。fixtureの再生成は完全一致し、
SHA-256は `f137302388b927150ffc000954e6e97e1ef60aba66c5fbedf74f42504685d1b3`。
合成結果を既存の共同正規化へ渡し、共有重みとall-unknownの質量も照合した。
同点でのID順序、入力順序変更、失敗後のscratch再利用、各成分の時間尺度、
単一stayのhop分割とunknown leakを追加検査した。これを全共同モデルの時間感度検査とはしない。

新規Rust検査5件。全Rustは1,127成功・失敗0・無視33（31 suite）、
`cargo test exit=0 @ 2026-09-18T06:52:10+09:00`。
Clippy・全target check・fmt・diff check通過。結果とsource hash、全テストログは
`target/i10-joint-proposals-20260918/` に保存した。

共同moduleは引き続きtest-onlyであり、通常観測のposteriorを置換していない。
次の接続単位は、実観測に支持された共有context候補と局所pathの所有・寿命を用意し、
登録した条件付き特徴・scoreを同じ観測から共同正規化へ通す経路である。
仮の共有context重み1やscore差0で、未実装の共同推論を代用しない。
全head、actual-Voice/default対応、評価時刻、全I10資源検査も残る。I9保留を維持する。


## 2026-09-18: 候補head内の音群別部分支持を保持

帰結核の各alternativeに渡すhead入力を、正規化したカテゴリ分布だけでなく、
そのalternative内で支持された音群質量とカテゴリ分布の組へ修正した。
以前の入力契約ではheadを全支持または全欠測のどちらかとして扱い、一部の音群だけが
支持された場合の質量を保持できなかった。登録は `feature-manifest.json` の
`i10_partial_head_support`。

既存の `WeightedRating` を集計結果にも使い、`ratings::mix_rating` が支持された音群・pathの
カテゴリを混合する。未支持・未列挙の重みは支持側へ配分しない。この段階ではtemperatureと
priorを適用しない。共同帰結側は保持質量、仮想的な候補重み、そのheadの音群支持質量を掛けて
候補間を混合する。その後に一度だけtemperatureを適用し、固定したhead別の発行時coverageと
合計支持質量でpriorへbackoffする。通常のraw head診断も同じ集計関数を経由する。

headの欠測だけでは解釈候補を捨てず、候補の重み・entropy・解釈unknownは変更しない。
元からのunknown、帰結score未支持、head未支持の境界を維持する。
共同posteriorからのpreview adapterでも、異なるhead支持0.15／0.025／0と、異なる
coverageを通した結果を確認した。支持質量の非有限値・負値・1超過は拒否する。

独立Decimal・80桁の参照は、Rustの二段集計に合わせて作らず、音群と候補を一度に
確率空間で足し合わせた。異なる音群支持、欠測、全欠測、coverageゼロ、非単位temperatureを
含む10ケースと一致した。fixtureのSHA-256は
`9ac7986febd156691e58a96994d54a69d9778357b50d682aefdef31ae1c1530d`。
既存60ケースのfixtureは完全に同一のまま通過し、両fixtureの再生成も一致した。

通常rendererでは、前段と同じ持続音・振幅変更・三身体の3素材を再取得した。
両busで1,618 cell・3,236分布を独立照合した。複数音群の支持を持つ28件、
全量prior 1,200件、全窓が未来の912件を含む。
変更前と同じcutで比較できた通常観測261件、3素材の両bus終端、onset 5件、
身体default 405件、body descriptor 636件、自声outcome／descriptor予測各10件、
population step 2,009件とWAVが一致した。reportの非同期間引きにより共通cutだけを比較し、
全配信列の一致とはしていない。

全Rustは1,128成功・失敗0・無視33（31 suite）、
`cargo test exit=0 @ 2026-09-18T07:02:22+09:00`。
Clippy・全target check・fmt・diff check通過。実行command、独立照合、source hash、
全Rustログは `target/i10-partial-head-support-20260918/` に保存した。

通常観測のgesture・phrase・sectionは依然として独立診断の更新であり、その周辺分布を
共同事後分布として接続していない。今回閉じたのはhead部分支持の数値入力契約と共通集計である。
実際の共有context／局所pathの所有・寿命、条件付き特徴・score、残るhead入力、
actual-Voice/defaultと評価時刻、全資源検査は引き続き未完。I9保留とI11作用無効を維持する。


## 2026-09-18: 候補accent密度をcontinuation入力へ接続

候補窓のaccent密度を観測prefixだけから計算していた経路を拡張した。
保存済み身体profileのrise／fluxと、身体modelに固定されたaccent標準化値から、
通常観測と共通のsalience・peak判定を使って候補accentを求める。
登録は `feature-manifest.json` の `i10_projected_accent_density`。
観測側のsalienceとpeak条件の算術は維持した。

4個の連続した支持付きhopと3個のsalienceを必要とし、middleがthresholdを超え、
leftより大きくright以上の場合だけ採用する。したがって同じ高さが続くpeakは最初だけを採る。
4-hop全体が評価時刻までに完結する必要がある。途中で切れた最後のhopを再解析したことにはせず、
右側の候補入力を先読みしない。発行境界のaccentが未来のright contextで初めて判定可能に
なった場合もprojectedとして扱う。profile接続直後のflux欠測は維持する。

観測済みaccentと、候補入力に依存したaccentの重みを別集計する。
分母は観測済み物理支持と、未来の判定可能なmiddle-hop区間の合計で、重複させない。
ずらした2秒窓に90%以上の支持がある場合だけ密度を供給する。
支持付きゼロ、欠測、projected値を区別し、支持sample数・重み・候補accent数・最後の区間を
reportへ出す。UIにはprojected密度を持つcell数を追加した。
continuation headの既存密度座標へ接続し、観測accent ledger、周期推定、arrival reset、
grouping、episode記憶には書き込まない。

Rustの新規3検査で正のpeak、plateauのtie、発行境界、未完右側hop、欠測energy／flux、
時刻順序・出所の不正、元の観測ledger不変を検査した。
実音の持続音・振幅変更素材は、診断用に密度係数0.25・密度欠測係数-0.5を設定し、
同じ設定の変更前binaryと変更後binaryで比較した。これらは適合済み・採用済み係数ではない。

両busで1,576 cell・3,152カテゴリ分布を独立照合した。
観測detector履歴が不要な完全未来窓888件について、profile binaryから別計算した
4-hop支持・peak・密度と一致した。この888件はすべて支持付きゼロで、正の候補accentは
この実音比較では検査できていない。正のpeakの証拠は前述の合成fixtureに限る。
同じcutの候補cell 1,562件を比較し、密度が欠測からprojectedへ変わった902件、
pathカテゴリ分布の変化13,530件、raw混合分布の変化886件を確認した。
最大カテゴリ差はpathで0.122459、raw混合で0.032766。
これらは入力の接続と欠測処理の効果であり、予測精度・可聴性の合格とはしない。

通常観測の共通cut232件と両素材の両bus終端、onset 2件、身体default 253件、
body descriptor 252件、自声outcome／descriptor予測各4件、population step 1,259件と
WAVが変更前後で一致した。全Rustは1,131成功・失敗0・無視33（31 suite）、
`cargo test exit=0 @ 2026-09-18T07:16:05+09:00`。
Clippy・全target check・fmt・diff check通過。command、比較前binary、独立検証結果、
source hash、全Rustログは `target/i10-projected-accent-density-20260918/` に保存した。

I10全体は未完了。候補accentを用いた残るtiming／invitation、実共同状態と条件付きscore、
残るhead入力、actual-Voice/default対応、評価時刻と全資源検査を残す。
通常表の評価時刻は従来どおりaction時刻であり、onset／release後の音を窓へ入れる変更は
行っていない。I9保留とI11作用無効を維持する。


## 2026-09-18: 候補arrivalをcontinuation入力へ接続

実音群のarrival Engineを、bus・epoch・generationと発行cutが一致する場合だけ凍結する。
Recurrenceの退役・新生・失敗・古いcutは入力にしない。通常Tableは発行時の係数、観測accent、
間隔比、period・word文脈を保持し、各候補には独立した固定長scratchを渡す。
原Engine、観測accent ledger、period推定器、memoryの更新は行わない。

候補accent密度と同じ4-hop検出を使い、中心区間と右側hopの利用可能時刻をscratchへ渡す。
支持された候補accentでのみ経過時間をリセットする。欠測または区間の飛びは間隔比を消去し、
resetを未知にする。後の候補accentは候補側で新たに生じた未知だけを解消できる。
発行時点で既に未知だったresetは解消しない。評価時刻まで完全なhop支持が届かない端数窓も
未知のままとし、部分PCMの再解析や既知無音への置換を行わない。

通常arrivalと同一のhazard／periodic式から条件付き点予測を計算し、1秒horizonの場合だけ
continuationのarrival入力に渡す。reportは元のanchor、候補anchor、元と候補の未知状態、
処理済み時刻、端数窓、数値評価数を分離する。UIはprojected arrival cell数を表示する。
通常表の評価時刻はaction時刻のまま。校正済みquantileや新しい生成作用は導入していない。

追加6試験は、定数hazardの閉形式、時間依存hazardの解析積分、periodicのreset差、
元の未知の保存、区間欠落、端数窓、不正入力の原子的拒否、発行identity、
合成profileの候補accentからcontinuation分布までを検査する。
既存の実NSGT pulse試験にも、通常forecastと凍結状態の一致、古いcutの拒否を追加した。

通常rendererはsustain・stairに加え、pulse・held-pulse・attack-returnを取得した。
全素材は固定body modelとprofileを維持し、arrival horizonを1秒にした。
continuationのarrival係数1、対応する欠測係数-0.5は接続検査用の未適合値である。
sustain・stairの候補1,576 cellとraw混合3,152分布を独立照合した。
五素材の共通cut501件、両bus終端10件、既存onset／身体default／descriptor／outcome／
population記録とWAVは変更前後で一致した。

pulseとattack-returnから合計112件の発行時arrival状態を凍結できたが、その音群は
固定8 medoidとの照合対象に入らなかった。したがって、この実音比較は候補点予測の
正例を証明しない。正の候補accentからarrival・continuationへ到達する証拠は合成fixtureに限る。
判定閾値やmedoidを緩めて正例にする変更は行っていない。
取得command、比較前binary、両側report／WAV、独立照合結果は
`target/i10-projected-arrival-20260918/` に保存した。

最終ソースの全Rustは1,137成功・失敗0・無視33（31 suite）、
`cargo test exit=0 @ 2026-09-18T07:40:35+09:00`。Clippy・全target check・fmt通過。
全出力と終了状態も同じartifact directoryに保存した。
I10全体は未完了。通常の共有proposal・path寿命・条件付き特徴とscore更新、
残るtiming／invitationと全head、actual-Voice/default対応、評価時刻の確定、全資源検査が残る。
実音上の正例、予測精度、校正、可聴性・作者採用も未証明。I9保留とI11作用無効を維持する。


## 2026-09-18: articulation proposalを通常観測更新と共用

共有・局所の共同状態とscoreを接続する前に、既存の通常Gestureと共同tuple用の
articulation proposalが別実装になっている境界を解消した。
`gesture::articulation_proposals`が既知状態の遷移候補、既知に残る質量のlog値、
未知への移行質量を返し、通常の既知parent／未知parent更新と
`joint::proposals::List::articulation`が同じ核を使う。

新しいgapは観測支持付きの低energy入力がある場合に限る。欠測でも既知gapのstayは残る。
未知parentは新しい観測支持がある場合だけ既知状態へ入れ、既知候補なしでは未知質量1を保つ。
10秒の物理時間定数を一度だけ適用し、その後の枝刈り質量へ再適用しない。
通常更新は従来の線形積算順序を保ち、共同tuple側はlog表現で候補を並べる。
`exp(log_keep)`が1に丸まる極小時間差では`expm1`により未知への移行を失わず、
長い時間差では線形値がunderflowしても有限なlog survivalを保持する。

最初の実装では通常更新にもlog→線形の往復を導入したが、丸め差がbeam枝刈りを通じて
増幅され、実音二素材のarticulation状態に最大0.002373385898の差が出た。
この版は採用せず、比較binaryとreportをartifact directoryの`log-roundtrip/`へ保存した。
判定許容誤差は広げず、通常更新の乗算・除算順序を戻すことで解消した。

追加2試験では通常Group更新36条件を独立した閉形式と照合し、既知／未知parent、
4状態、欠測・通常energy・低energy、1／100／32,000 msを通した。
極小の未知移行、長時間のlog保持、不正な物理時間の拒否も検査した。
既存のjoint proposal 5試験（12ケースの独立Decimal全列挙を含む）も通過した。

修正後の通常rendererはsustain・stairの共通218 cutで、delivery時刻計測値を除く
観測snapshot全体と候補表が数値差ゼロ、両bus終端とWAVも一致した。
onset 2件、身体default 253件、outcome／descriptor予測各4件、population step 1,259件も一致。
身体descriptor 252件はpayloadが一致したが、sustainの両bus各1記録はavailableが
128,000から128,512へ1 hop遅れた。元のsource endは128,000のままで、未来情報はない。
この到着遅延差を消した扱いにはせず、実時間・全資源の合格も宣言しない。

取得command、比較binary、両側report／WAV、検証scriptと結果は
`target/i10-live-articulation-proposals-20260918/` に保存した。
全Rustは1,139成功・失敗0・無視33（31 suite）、
`cargo test exit=0 @ 2026-09-18T07:54:15+09:00`。Clippy・全target check・fmt・diff check通過。
全Rust出力、終了状態、最終source hashも同じdirectoryへ保存した。

これはarticulation成分の通常接続であり、共同posteriorの実装完了ではない。
共同正規化とconsequence核は依然test-only。共有proposal、実共有・局所pathの寿命、
完全な条件付き特徴座標とscore、他成分の通常接続、全head、actual-Voice/default、
評価時刻と全資源検査は未完。I9保留・I11作用無効を維持する。


## 2026-09-18: phrase親参照の拒否を状態更新前へ移動

実際のphrase→section消費では、childの`parent_index`は直前の特定beamを参照する。
既存実装も`previous_end_sample`の一致を検査していたが、その前にcueを更新し、
公開snapshotのgroupsを消去していた。途中のreceiptが抜けた入力は拒否されても状態が
部分更新され、抜けたreceiptを正しい順序で渡し直す経路を壊していた。
これは親時刻チェックの欠如ではなく、検査と更新の順序の問題である。

sectionの入口で、sample rate、acousticとphraseのend、censored状態、bus／epoch／
音群generation、音群の重複、直前beamの時刻、child配列の連続性と親index範囲を
全音群について検査する。cue消費、音群退役、context割当を始める前に拒否する。
既存の実音欠測を一つのphrase receipt内で進める親参照合成は維持する。
生成済みreceiptの配送抜けを、実音の欠測と読み替えない。

追加2試験は、実Phraseから作ったreceiptに対する24件の不正envelopeと4件の再送拒否、
有効receiptへの復帰を検査する。もう一つは常に1 path・親index 0の入力から
2番目のreceiptだけを抜き、3番目の拒否で状態が変わらないこと、2番目→3番目の再投入で
処理が進むことを確認する。新しい先行検査を外した版では、拒否後のsnapshot一致検査が
実際に失敗した。そのログを保存し、最終sourceへ戻した。
この変更で全種類の実行途中エラーまで原子的になったとは主張しない。

通常renderer比較は、sustain・stairに明示的なtemporal_section設定を追加して行った。
section候補と各素材で最大7件の確定phraseが実際に出る条件で、共通230 cutのsnapshot全体・
候補表が数値差ゼロ、両bus終端とWAVも一致した。delivery delay計測値は比較から除いた。
onset 2件、身体default 253件、outcome／descriptor予測各4件、population step 1,259件も一致。
身体descriptor 252件のpayloadは一致し、stairの両bus各1記録だけavailableが
15,360から15,872へ1 hop遅れた。両側ともsource end以降の時刻であり、遅延差は別に残す。
実時間・全資源の合格はこの比較から宣言しない。

比較binary、command、設定、report／WAV、旧版の失敗再現、検証scriptと結果は
`target/i10-parent-receipts-20260918/` に保存した。
全Rustは1,141成功・失敗0・無視33（31 suite）、
`cargo test exit=0 @ 2026-09-18T08:06:08+09:00`。Clippy・全target check・fmt・diff check通過。
全Rust出力、終了状態、最終source hashも同じdirectoryへ保存した。

共同状態の親子参照に必要な実消費者の境界を強化したが、共有・局所joint pathの生成・
正規化を通常経路へ接続したわけではない。共同特徴の全座標とscore、全head、
actual-Voice/default、評価時刻、全資源検査は未完。I9保留・I11作用無効を維持する。


### 2026-09-18: 発行時の実Voice時計入力を候補時刻診断へ接続

従来のdefault診断は受理済み指令・活動・予約Toneだけを参照していた。
今回は発行時の時計状態を実Voiceから通常rendererへ渡し、未予約の点も最大16点の
候補時刻診断へ接続した。default対応全体の完了ではない。

`Opportunity` は発行時刻・対象時刻と四種類の出所を持つ。Participationでは既に保持する
`due_frame` または `planned` を読む。初期化前・過去の点は未知とし、観測側でpolicyを
初期化、追いつき更新、選択、resolveしない。Couplingでは現在の位相・実速度、Thetaでは
現在の位相・周波数と既処理gateから次の交差を投影する。Couplingの速度計算は通常時計と
共用し、乱数を引かない。後二者は現在のrhythm入力を固定した投影であり、将来のrhythm変化や
発音を保証しない。Participationのdueも、その後のreference更新や選択で変更され得る。

発行前policyと発行後の更新済み周期が混在していたため、`PhonationEngine::tick` 前に
時計点とintrinsic periodを凍結するよう揃えた。Holdでは離散機会を渡さず、再利用batchの
時計点・周期もclearする。生存・gate閉鎖は別のpolicy事実として保存し、時計点から
発音許可、wait、skip、無音の帰結を生成しない。公開型は、既に外部exampleが使う
`PhonationBatch` のpayloadとして必要な範囲に留めた。

通常の`body_default` recordをversion 2へ更新した。既存の受理指令／活動／予約による
`default_input` があればその時刻を優先し、なければ時計点を時刻列の保持点に使う。
両者ともなければ列は空。spanは既知の発行時周期に対する`min(2P,4s)`で、12個の均等点を
加えて重複除去・整列する。周期不明では保持点だけを残し、保持点はspan外でも捨てない。
固定長配列の有効範囲は`candidate_time_count`まで。安定した帰結表の残horizonと校正済み
arrival分位点は未接続。reportと最新recordのUIへ渡し、未来classの適格性や実行へは使わない。

4件のRust検査を追加した。Theta、自由／結合Couplingの実交差と凍結した最初の点を照合し、
反復読み取りあり／なしで後続candidate列・gate番号が一致した。Participationではdueから
plannedへの移行と20回のrenewalを通常選択と照合した。実Voiceのbatch→Bankでは
renewal前の周期と時計点を保持し、Hold切替後に古い値が残らないことを確認した。
別のBank検査で閉じたgate、範囲外の保持点、過去issueの拒否、活動defaultの優先を検査した。
初回all-target checkで新payloadを持たない明示的batch initializerを検出し、exampleと
既存fixtureへ未知値を明示して修正した。

実音素材は `target/i10-clock-inputs-20260918/` に保存した。sustain、pulse、flowと
無発音entrained Voiceの組合せを、新旧binaryで同じconfigから描画した。既存のsection入力も
有効にした。`run.py before` / `run.py after` と `verify.py` で再現できる。
463件のdefault記録について候補時刻を独立再計算した。時計点はTheta 57件、Participation due
194件、planned 32件。時計点を持つが既存default actionは未知の記録156件、gate閉鎖中の
時計点113件を保持した。Couplingの新旧実音report比較はこの三素材に含めず、実時計・実Voiceの
Rust検査で確認した。これらは将来の全機会を捕捉する主張ではなく、既存20 Hz採取の結果である。

両busの共通258観測点で、候補表を含む観測の数値差は0。各busの終端も比較した。
非同期delivery delayを除外し、採取数が違う観測全体の一致とは扱わない。
受理済みonset、身体descriptor、self-sound outcome、descriptor prediction、population stepと、
新規field／versionを除いた既存default記録は一致した。今回descriptorの公開時刻差も0。
3素材のWAV hashは新旧でそれぞれ一致した。新しいflow素材のSHA256は
`fd9e9373de6d6eef80866d76cdf185e22c4de838fa4bec2a5b0214cbb279a4c5`。

全Rustは1,145成功・失敗0・無視33（31 suite）。
`cargo test exit=0 @ 2026-09-18T08:18:54+09:00`。
通常Clippy、全target checkも通過した。時計queryはgated Voiceごと毎hopに加わる。
Bankは既存64枠を使い、新しいheap確保はないが、16時刻配列、出所、コピー、reportと
queryの費用を全I10/R2測定へ算入する。全資源検査の合格にはしない。

I10全体は未完了。今回の点は未来actionの確約ではなく、明示的な再考要求と
七classの未来適格性・身体profile対応、actual/defaultの音響paired差分は残る。
通常の共同状態・path寿命・条件付き特徴／score、全head、評価時刻、全資源検査も残す。
I9保留・I11作用無効を維持する。


### 2026-09-18: 実身体へのdescriptor-only転用を反例で棄却

actual-Voice/default接続の前提として、保存済みの実身体396分岐と固定8-medoid profileを
照合した。`scripts/evaluate_temporal_profile_transfer.py` を追加し、入力hash、照合規則、
評価窓、未知の扱いを `target/i10-actual-profile-transfer-20260918/registration.json` に固定した。
これは既に調べた素材を使う探索的development監査であり、held-out、適合、新しい合格閾値は
導入していない。通常の共有profile表・生成方策・音声には変更を加えていない。

入力は36条件・396分岐・両busの既存実音特徴（792比較）、固定binary profile、凍結した
body modelである。発行までに利用可能な最新の私有descriptorだけを使い、既存の標準化、
距離<=0.25、共通座標数、medoid ID順のtie規則で割り当てた。候補時刻には最寄りwhole cellを
使い、中点tieは前方、未知cellの代替探索はしない。未来frameの第一hop rise/declineは
実前史のlog RMSから計算し、第一hop fluxは未知、背景依存のenergy shareは比較から除いた。
0.25/1/2/4秒のsource energyは、hop平均に物理窓との交差幅を掛けて積分した。
これは仮想帰結の評価時刻を通常headで変更したものではない。

72条件（36条件×2 bus）すべてが距離条件を通った。発行前に利用可能な最新descriptorは
いずれもmask=44の三座標で、centroidなどを持たない。うち24条件では、実際にはroutedな
busが、元busには出力しないmedoidへ対応した。この対応ではonset後もprototypeは無音となる。
例えば `sine-both-silent-gate` のbus 0、即時onsetは、実音の1秒energy積分が
0.002004367968844844、profile予測が0だった。descriptor一致は候補発音の適合を保証しない。

活動身体でも順序反転が出た。`sine-both-two_tones-gate` のbus 0で、1秒energy積分は
continue=0.0026111455603239618、0.5秒後onset=0.003823159110767009で後者が大きい。
profileではcontinue=0.0008083112955623099、最寄り0.516125秒cellのonset=
0.0004459058386012331で逆転した。32-cell近似、異なるpitch/envelope/前史、干渉などの
誤差をこの一比較から個別に分離したとは主張しない。

各窓で4,104組の同一条件内candidate比較を行い、0.25/1/2/4秒の反転はそれぞれ
280/356/304/208組、実差を予測上のtieへ潰した組は360/504/504/504だった。
これはsource energyの順位であり、ordinal評定や望ましい音の順位ではない。
両側が無音の対照も母数に含め、支持のある都合のよい部分だけに絞っていない。

監査初版では、活動条件のcontinue・無活動条件のskipをdefaultと名付けていた。
素材はその実policyのdefault選択を収録していないため、この呼称を訂正した。
数値対照は同じ無追加指令条件のまま維持し、`control_class` として記録した。
初版registration/resultも保存した。実際のbody-defaultの選択・将来指令列とのpaired比較は
依然未取得であり、この無追加指令比較で代用しない。

別のscalar verifierは、元rendererの `energy_per_hop` とbinaryの96-byte frameを直接読み、
72割当、792両bus分岐の積分・対照差、288条件別順位集計を独立照合した。
energy、cell、assignmentを故意に変えた3負例はすべて拒否した。
`tests/test_temporal_profile_transfer.py` の2検査も通過し、入力hash変更・未登録入力の拒否、
中点tie・horizon外・未知cellの保持を確認した。Pythonコンパイル検査とdiff checkも通過。
`src/`変更はなく、既存Rust検査の再実行は行っていない。

再現:

```bash
python3 scripts/evaluate_temporal_profile_transfer.py \
  target/i10-actual-profile-transfer-20260918/registration.json \
  target/i10-actual-profile-transfer-20260918/result.json
python3 target/i10-actual-profile-transfer-20260918/verify.py
python3 -m unittest discover -s tests -p test_temporal_profile_transfer.py -v
```

結果はdescriptor-onlyの実Voice転用を採用できない根拠として扱う。距離閾値を緩めたり、
無音busのmedoidを別の都合のよい候補へ差し替えたりして合格にしない。次の転用モデルには、
source routing、実身体のpitch/envelope/rateと前史・既存Toneの条件、実policy defaultの取得を
明示し、developmentと新しいheld-out比較を分離する必要がある。通常のraw prototype診断は
既存の限定範囲を維持し、ordinal帰結やactual-Voice action eligibilityへ昇格させない。
I10全体は未完。共同状態・条件付きscore、全head、評価時刻、全資源検査も残す。
I9保留・I11作用無効を維持する。


### 2026-09-18: 実policyのdefault指令列・音声を取得し、保存入力から再生

前監査で不足を確認した実policy defaultの取得を実装した。
`src/runtime/body_profiles/policy_default.rs` は通常workerを実行し、一Voiceの全batch、
ToneSpec、policy／時計点、身体世代とeffective body条件、rhythm入力、guard前の両bus音声を
保存する。六つの新しいdevelopment素材とconfigを取得前にhash固定した。
`target/i10-policy-default-inputs-20260918/registration.json` が入力登録、
`target/i10-policy-default-20260918/` が完成した取得結果である。

素材はsine hold／flow／無発音entrained、harmonic entrained／pulse、modal flow。
両bus、habitatのみ、presentationのみを含む。48 kHz／512 sample hopで、発行を24,576 sample、
未来を192,000 sample（4秒）に固定した。各素材は実Voice一つに限定し、renderer出力を
そのVoiceの私有音声として扱える条件を明示的に検査した。発行前48 hopと未来375 hopの
合計423 hopを保存し、6素材合計2,538 hopとなった。

実defaultは、通常workerがその後に発行した指令列そのものとした。未来入力には
「発行時点で利用可能な予測入力ではなく、後で観測した教師」と明記した。
発行時のrendererをsource単位で複製した経路と、空rendererから保存JSONのprefix・未来指令を
復元する経路の二つで、実defaultの音声を再生した。前史のbackend／envelope状態、実routing、
時計と全指令の対応を引き継ぎ、両経路の音声を通常worker出力と各hopで完全一致させた。
将来の全指令を発行時に知っていたことにはせず、実policyそのものをJSONから復元したとも
主張しない。保存した実policy出力をrendererへ再投入した検査である。

比較用に、発行後は指令を一切追加しない経路も別保存した。rhythm入力は実defaultと共通の
保存済み系列を使う。これは無追加指令の対照であり、defaultや介入後の閉ループpolicy応答の
代用ではない。実defaultの未来On/Off/Update数は合計34/30/0。6素材中4素材で無追加指令と
音声が異なり、両bus合計1,099,959 sampleで値が変わった。Holdと無発音素材は一致した。
この差は旧素材のcontinue／skipを実defaultへ読み替えられない具体例でもある。

ToneSpec、身体・modulator、指令、routing、rhythm、batchの直列化／復元は既存型への
`cfg(test)` deriveで実装した。Arc配列用のSerde rcもdev dependencyに限定した。
新しい取得器もtest build限定で、instrumentの通常binaryに音声保存機能は追加していない。
身体の条件をDebug文字列へ潰さず、後続の転用検査が読める構造化データとして残す。

初回はRustでの再生が一致した一方、別verifierがissue metadataとtraceの数値表記差を検出した。
直接Serializeしたf32の短い表記と、JSON Valueへ変換したf64表記が混在していた。
traceもValue経由で保存するよう揃えて再取得し、厳密なmetadata比較を通した。
初回のデータは `target/i10-policy-default-20260918-initial/`、失敗ログも入力directoryに保存。
新旧PCM 60ファイルのSHA-256はすべて一致し、数値表記の修正が音声を変えていないことも確認した。

`scripts/verify_temporal_policy_defaults.py` は、登録hash、423 hopの連続性、発行時入力、
source／世代／routing、policy／opportunity時刻、最初の未来onsetを照合した。
通常reportの70件のOn/Off受領記録と保存指令のsource・時刻が一致した。
各busの実default／JSON再生／状態複製再生はPCMのSHA-256が同一。
全未来hopのenergyを保存PCMから独立再計算し、非routing busの無音と、対照との差の件数も
確認した。issue時刻、最初のonset、routing、replay PCM、energyを改変した5負例はすべて拒否した。

再現（出力directoryは未作成のパスを使う）:

```bash
CONCHORDAL_I10_DEFAULT_INPUTS=target/i10-policy-default-inputs-20260918 \
CONCHORDAL_I10_DEFAULT_OUTPUT=target/i10-policy-default-20260918 \
RUST_BACKTRACE=1 cargo test --lib \
  runtime::body_profiles::policy_default::acquire_policy_defaults -- --ignored --exact --nocapture
python3 scripts/verify_temporal_policy_defaults.py \
  target/i10-policy-default-20260918 target/i10-policy-default-inputs-20260918
```

取得試験1件成功（17.73秒）。通常Rustは1,145成功・失敗0・無視34（31 suite）、
`cargo test exit=0 @ 2026-09-18T08:44:31+09:00`。
通常Clippy、全target check、fmt、diff check、Pythonコンパイル検査も通過した。
全Rust出力・終了状態、binary／source hash、取得・検証ログを入力directoryへ保存した。

ここで取得したのは実defaultの教師と再生可能な身体条件である。これらのdefaultと同じ発行時
状態に七classを介入する素材、body-conditionedな転用モデル、新しいheld-out比較、通常経路の
未来適格性とactual/default帰結差分は未完。新しい取得で未知だった未来を発行時入力へ混ぜない。
I10全体の共同状態・条件付きscore、全head、評価時刻、資源検査も引き続き必要。
I9保留・I11作用無効を維持する。

### 発行時に既知のrelease予約と局所身体分岐（2026-09-18）

対象はI10のactual-Voice候補入力である。前段の無追加指令対照は、PhonationEngineに
残っているnote-off予約まで止めてしまうため、既存身体のcontinueを代用できない。
今回はその予約を発行前に凍結し、実身体のcontinue／release／gap分岐へ渡した。
終了点は局所音響分岐と独立照合までであり、通常の候補選択や全I10の完成ではない。
明示取得は18.22秒。実装時間と後続実装の残工数は未計測。

`target/i10-local-continuation-inputs-20260918/registration.json` に取得前の条件を保存した。
前段と同じ6素材・seed・config、48 kHz／512 hop、発行点24,576 sample、未来192,000 sampleを使う。
変更はtest build内の取得経路だけであり、通常instrumentに録音経路や新しい生成作用は加えていない。

`PhonationEngine::planned_releases` は未送出の予約heapを読み取る。発行直前hop終了時の
実renderer、toneごとのenvelope、予約release、更新後のintrinsic period、最後のrhythmを凍結する。
予約とenvelopeをJSONへ保存して読み戻した上で、両busの局所実音を生成する。
未来の実policy指令と未来の観測rhythmは、この分岐生成関数へ渡さない。
rhythmは最後の状態からsample単位で外挿する。実policy未来は別の教師として保持する。

continueは新しいexcitationを加えず、発行時に予約済みのreleaseを元の予定hopで送る。
予約を未来時刻付きのまま早く送ると、短いholdではattack長にも影響するため、早期送出しない。
release／gapは0、6,000、24,000、48,000、96,000 sample後の5時刻を比較する。
既知の予約で更新したenvelopeの `onset <= at < release_end` を満たすtoneだけをreleaseする。
gapは現在周期を固定したwithholding endpointと自然tailを保持する。周期が未知なら候補を作らない。
この素材には未開始の予約onsetはない。検査器はそれが混入した取得を拒否する。
予約onsetの取消しを含む一般的なgapの成立は、この取得から主張しない。

共通の局所previewは後続の新しいexcitationを仮定しない。gapの周期終了後に必ず無音が続く、
あるいは必ず再発音するという方策予測ではない。`planned-control` は既存身体の予定軌道であり、
実policyの未来列全体とは区別する。既存身体がinactiveならcontinue候補にも分類しない。
`unplanned-control` は同じ凍結rhythmで予約releaseだけを脱落させた対照である。

取得結果は `target/i10-local-continuation-20260918/` に保存した。

- 実機関内の予約releaseは5件。局所候補44件と対照12件、合計56分岐を取得した。
- 実状態のforkと、保存済み全prefixを再生して復元したrendererの出力は、全分岐・両busでsample単位に一致した。
- 22候補を除外した。自然終端後のrelease、inactive bodyのcontinue、周期未知のgapを含む。
- release予約を落とした対照との差は4素材、両bus合計934,446 sampleだった。
  harmonic-entrainedは発行時に52,527 sampleのreleaseを予約しており、自然終端は70,287 sample。
  発行点から48,000 sample後（絶対時刻72,576）のrelease候補は既にinactiveであり、除外された。
- 旧policy取得のPCM60ファイルは全てSHA-256一致。未来指令34 On／30 Off／0 Updateも不変だった。

`scripts/verify_temporal_local_continuation.py` は元のdefault検査に加え、保存した予約と実Off receipt、
envelope終端、候補集合・活動・class入力、送出指令、routing、物理energy、自然終端以降の無音を独立照合する。
continueと予定軌道、同時刻のreleaseとgapのPCM一致、介入hopより前の不変性も検査した。
未開始onset、予約時刻、自然終端、終端後releaseの許可、gap endpoint、energy、終端後PCMの7変異を全て拒否した。
変異後は元ファイルを復元して再検査した。

再現コマンド:

```bash
CONCHORDAL_I10_DEFAULT_INPUTS=target/i10-local-continuation-inputs-20260918 \
CONCHORDAL_I10_DEFAULT_OUTPUT=<fresh-output-directory> \
RUST_BACKTRACE=1 cargo test --lib \
  runtime::body_profiles::policy_default::acquire_policy_defaults -- --ignored --exact --nocapture
python3 scripts/verify_temporal_local_continuation.py <fresh-output-directory> \
  target/i10-local-continuation-inputs-20260918 target/i10-policy-default-20260918
```

明示的取得は1件成功（18.22秒）。通常全体テストは1,145成功、0失敗、34 ignored、31 suite。
`cargo test exit=0 @ 2026-09-18T09:00:36+09:00`。通常Clippy、全target check、fmt、diff検査も通過した。
ログ・status・検査対象source／test binaryのhashは入力artifactディレクトリに保存した。

未接続は、消費前の実機会と一つの固定recipeを伴うonset／delay／wait／skip、通常の候補許可・
default対応・資源上限、実身体転用の適合と独立検証である。今回の固定prefixにおける音響介入を、
実policyの閉ループ反実仮想やlive decisionの許可へ読み替えない。通常共同状態・条件付きscore、
全head、評価時刻、全資源検査も残る。I10全体は未完了。I9保留・I11作用無効を維持する。

### 実機会と一つの発音recipeの通常接続（2026-09-18）

対象はI10の残るonset／delay／wait／skipへの実身体入力である。従来のhop冒頭の時計予測点だけでは、
実際に選ばれた機会か、既に消費された機会か、元のintrinsic dueから移動した機会かを区別できなかった。
今回は実時計が選んだcandidateと、そのOnに対応して一度だけ作るToneSpecを結び、通常report／UIまで通した。
終了点はこのproducerと実消費者の接続・照合であり、4 classの音響分岐や実行許可の完成ではない。
実装時間と残工数は未計測。明示的な実音取得は18.13秒だった。

`OnsetOpportunity` は発行hop、選択時刻、gate、元のintrinsic due、既知の身体周期、予約releaseを保持する。
Participationの元dueは、`OnsetRule`の処理とclockの`resolve`より前に実policyから読み取る。
grantが出た場合だけToneOnEventへ保持し、同じ処理で確定したrelease時刻を添付する。
Voiceは通常一回の身体・pitch・ADSR・render modulatorからToneSpecを作り、receiptをそのまま添付する。
追加の音色候補、身体乱数の再抽選、未来の実音からの許可推定は行わない。

Holdには周期機会を付けない。ThetaとCouplingでは実crossingを保持するが、周期は未知とする。
Thetaのgate周波数はaccumulator等を含む身体の発音周期ではなく、Couplingではgather後のrateが
既に次のrenewalを表す場合があるためである。未来の周期や時刻を代入して穴埋めしない。
手製のexample/test recipeにはreceiptを付けず、medoid取得器が実recipeを別時刻へ移すときも消去する。

`body_default` version 3は、当該hopのpolicyがalive／gate openであり、実On指令のtone IDと
materialized recipeのID・発行時刻・onset時刻が一致したreceiptだけを公開する。
記録するのは最初の対応receipt、recipe ID、対応件数であり、通常20 Hzの診断間引きは維持する。
この出力は過去のgrantの診断である。report到達時点で未消費の機会が残っているという許可ではない。
UIにはrecipe ID、選択時刻、元due、予約releaseを表示する。

取得前登録は `target/i10-onset-receipts-inputs-20260918/registration.json`、結果は
`target/i10-onset-receipts-20260918/`。前段の6素材・seed・configを固定して再取得した。

- Gated Voiceから39件の実recipeを取得。Participationの30件は、選択より前のhopに保存された
  intrinsic dueと一致した。9件のTheta recipeは身体周期を未知のまま保持した。
- 通常report／UIのサンプル9件で、recipe IDとreceiptが保存した実ToneSpecに一致した。
- 新しいreceiptを除いた全producer入力・recipeパラメータは前段と完全一致した。
- 元policyのPCM60ファイルと既存continue／release／gapのPCM112ファイル、合計172ファイルのSHA-256が一致した。
- 共通のbus／source epoch／frame IDを持つ531観測は、delivery latencyを除く全状態・特徴が一致した。
  reportが選んだ非同期の最新frameは一部異なるため、未対応frame数を別途記録した。

最初の検査器はreport行単位で処理時間まで同値を要求し、`hop_timing/elapsed_us`で失敗した。
`verification-initial.log`を保存し、元の取得前登録は変更していない。
`comparison-followup.json`に、明示したwall-clock計測値の除外と観測frame identityによる照合を記録した。
reportの件数・順序・time_sec、その他の旧フィールドはそのまま比較する。
観測系列全体の同値や実時間性能の合格へ読み替えない。

`scripts/verify_temporal_onset_receipts.py`は、実On、gate、予約Offの通常receipt、旧入力、
過去のdue、同一recipe、サンプル診断との対応を独立照合する。発行時刻・選択時刻・元due・予約release・
recipe pitch・公開recipe ID・実観測featureの7変異を全て拒否し、復元後の再検査も通過した。

Rust検査では、実Voiceのrecipe→通常診断の一往復と、不一致・指令欠落・gate閉鎖の5拒否条件を追加した。
既存の20 renewal比較で、選択前／選択中／resolve後の元due参照と時計・RNGの不変性を検査した。
既存の音響混雑fixtureでは、元due 10,500 sampleから遅らせた選択時刻を同一視しない点を検査した。
今回の6実音素材にはその移動の正例はなかったため、実素材の全状況で証明したとはしない。

再現コマンド:

```bash
CONCHORDAL_I10_DEFAULT_INPUTS=target/i10-onset-receipts-inputs-20260918 \
CONCHORDAL_I10_DEFAULT_OUTPUT=<fresh-output-directory> \
RUST_BACKTRACE=1 cargo test --lib \
  runtime::body_profiles::policy_default::acquire_policy_defaults -- --ignored --exact --nocapture
python3 scripts/verify_temporal_onset_receipts.py <fresh-output-directory> \
  target/i10-onset-receipts-inputs-20260918 target/i10-local-continuation-20260918
```

通常全体テストは1,146成功、0失敗、34 ignored、31 suite。
`cargo test exit=0 @ 2026-09-18T09:15:38+09:00`。明示取得1件、通常Clippy、全target check、fmt、diff検査も通過した。
ログ・status・source／test binary hashは入力artifactディレクトリに保存した。
通常経路に新しいheap collectionは加えていない。既存event／ToneSpec／Recordの保持サイズ増加、
元dueの取得計算、batch内recipeとOnの照合費用は全I10/R2の測定対象へ加える。

次は、このreceiptと固定recipeを使うonset／delay／wait／skipの実音分岐と、同じ機会のlocal default差分を作る。
機会消費・再考・20 Hzと共有budgetの通常接続、実身体転用の検証、共同状態・条件付きscore、
全head、評価時刻、全資源検査は引き続き未完。I9保留・I11作用無効を維持する。

### 実機会における7 class・local default差分と十生特徴（2026-09-18）

対象はI10のactual-Voice候補と同じ機会のlocal default比較である。前段で得た実grant／recipeを
使い、同じ既存身体と既知のreleaseから7 classを分岐させた。次にそのPCMを通常の私有Laneへ通した。
終了点は条件付き実音・物理energy差・十生特徴の取得と独立照合であり、通常予測器への採用ではない。
明示的な分岐取得は0.61秒。実装時間と後続のモデル接続の残工数は未計測。

分岐の取得前条件は `target/i10-onset-branches-inputs-20260918/registration.json` に固定した。
前段のsource 205ファイルをhash照合し、24,576 sample以降に発行された最初の実grantを対象にする。
この停止条件は現在までのproducer receiptだけで決まり、それ以降の指令やrhythmを分岐関数へ渡さない。
対象hopはOn指令一つだけであり、未開始の既存onsetもない。この範囲外は別の取得条件が必要である。
Holdと無音の2素材には記録区間内に該当grantがなかったため、機会やrecipeを補わず取得対象外とした。

残る4素材では、過去の全実指令を再生してrendererを復元し、過去のreceiptから未送出のreleaseを再構成する。
元のprefixは保存した実PCMと全sample一致した。現在のgrantに対応するToneSpecとkickを一つ固定し、
onset／delayed onsetでは発音時刻と、それに対応する予約releaseだけを同量移す。
pitch、amp、body、ADSR、smoothing、render modulator、tone ID、kickは変えず、身体の追加乱数抽選も行わない。
仮想Onへ過去の実grant receiptを付け直すことはしない。modalの初期化seedは通常rendererと同じく
仮想発音時刻に従う。時刻に依存する通常の初期化を、別の音色候補や特別なseedへ置き換えていない。

local defaultは当該実機会で選ばれたonset-nowである。その最初のhopは実演時のPCMと両busで完全一致した。
これは既存身体の予定releaseを含む一つの身体動作であり、以降の実policy未来列全体の再生とは異なる。
全分岐で後続の新しいexcitationを仮定せず、当該hopで既知のrhythmをsample単位で外挿する。
wait／skip／gapの機会消費・再考・withholdingは入力契約として保持し、live policy上では実行しない。
reconsiderationやgap終端の後に必ず無音が続くという閉ループ予測にも用いない。

candidate offsetは0、6,000、24,000、48,000、96,000 sample。実grantと既存envelopeに対する
primitive入力則で35 class-time／素材を検査し、140組中71組を取得、69組を除外した。
周期未知のTheta機会でgapを作らず、自然終端後のreleaseも除外する。

| class | 取得分岐数 |
|---|---:|
| onset-now | 4 |
| delayed onset | 16 |
| wait | 16 |
| skip | 4 |
| continue | 4 |
| release | 12 |
| gap | 15 |

判断sampleはそれぞれ34,935／28,527／29,098／48,041であり、対応hop境界から119／367／426／425 sample後にある。
部分hopを作り直さず、元の完全な512 sample hopで、判断時刻+4秒を覆う次の境界まで376 hopを生成した。
判断前のPCMは全分岐で保持し、0.25／1／2／4秒のmean-square窓は正確な判断sampleから積分する。
71分岐×2 bus×4窓＝568組のcandidate値・local default値・差分を保存した。
これは物理energyの差であり、好ましさやordinal帰結ではない。通常headの評価時刻の仕様変更も行っていない。

例としてsine-flowのhabitat、判断後1秒では、local defaultのmean-squareは0.0017904671304218205、
0.5秒delayed onsetは0.00034575611166252814だった。wait／skip／continueは同一の無追加excitation軌道を持ち、
同時刻のrelease／gapも同じ自然tailを保持した。機会消費と再考の入力値は音響同値とは別に照合した。

分岐結果は `target/i10-onset-branches-20260918/`。`scripts/verify_temporal_onset_branches.py` が、
元のToneSpec・ADSR・過去の実Offからprefix envelopeを独立再構成し、既知release、候補活動、全class-time、
固定recipe、予約offの移動、routing、前史、終端後の無音、物理energy、local default差分を照合した。
recipe pitch、retimed release、waitの機会消費、終端後release、未知周期の捏造、default差分、
判断前PCM、tail後PCMの8変異を全て拒否した。

続く特徴取得の条件は同じ入力ディレクトリの `feature-registration.json` に固定した。
既存 `src/temporal_cognition/body/action_targets.rs` と `scripts/collect_temporal_action_features.py` に
本corpusの入力形式を追加し、通常のprivate Lane、48 kHz／512 hop、NSGT nfft 2048、right／coherent、
55–8,000 Hz・96 bins/octave、未適合unit尺度を共用した。以前のactual／medoid形式は同じ分岐を維持する。

結果は `target/i10-onset-branch-features-20260918/`。prefixは544 hop、未来は53,392 hopである。
未来の533,920座標をPCMと保存したLog2Space spectrumから独立再計算し、最大絶対差は
4.973799150320701e-14だった。body publication 5,372件の時刻・支持・identity、音響同値62組、
candidate前の不変性72組、無routingの39 streamも照合した。
NSGT自体と六記述値の窓数値は共有実装であり、この検査から独立再実装済みとは主張しない。

各rowに判断前／判断をまたぐ／判断後の位置を記録した。最初のtarget hopには判断前の実PCMが含まれるため、
これを完全な未来のhopとして扱わない。case metadataには判断時刻と、最初の完全な判断後hop境界を残した。
元のwarmupとNSGT支持を継続し、partial PCMの再解析は行わない。
straddling hopを完全未来とする誤表示、log RMS、source support、warmup再開始の4変異を全て拒否した。

再現コマンド:

```bash
CONCHORDAL_I10_ONSET_INPUTS=target/i10-onset-branches-inputs-20260918 \
CONCHORDAL_I10_ONSET_OUTPUT=<fresh-branch-directory> \
RUST_BACKTRACE=1 cargo test --lib \
  runtime::body_profiles::policy_default::onset_branches::acquire_onset_branches -- --ignored --exact --nocapture
python3 scripts/verify_temporal_onset_branches.py <fresh-branch-directory> \
  target/i10-onset-branches-inputs-20260918
python3 scripts/collect_temporal_action_features.py --inputs <fresh-branch-directory> \
  --output <fresh-feature-directory> --collect
```

通常全体テストは1,146成功、0失敗、35 ignored、31 suite。
`cargo test exit=0 @ 2026-09-18T09:38:00+09:00`。明示取得2件、Python既存2検査、通常Clippy、
全target check、fmt、diff検査も通過した。必須ログ・statusと検査対象source／最終test binary hashは入力artifactに保存した。

I10全体は未完。今回のraw実音教師を通常の有界予測モデルへ渡すには、実recipe・routing・envelope・
周期・前史の条件付けと同一機会のdefault対応を実装し、descriptor距離だけの転用失敗を解消する必要がある。
通常の機会消費・再考・rate／共有budget、共同状態・path寿命・条件付きscore、全head、評価時刻、
全資源検査も残る。反実仮想PCMをlive listenerの証拠や学習済みの精度へ読み替えない。
I9保留・I11作用無効を維持する。

## 発行時に既知のrelease予約を通常自声予測へ接続（2026-09-18）

対象はI10の発行済みonsetに対する自声予測である。前節の実recipe receiptには既に
`planned_release_at` があったが、rendererは予測へ `prediction_parameters(None)` のみを渡し、
予約が短い場合にも既定60秒のholdを使っていた。今回の終了点は、発行時に既知の予約を
通常の固定energy priorと既存の終端特徴へ渡し、実音を変えずに検査することである。
共有8モデルによる反実仮想予測の転用失敗を、この修正だけで解決したとはしない。

入力と比較条件は `target/i10-scheduled-release-inputs-20260918/prediction-preflight.json` に固定した。
通常forecastに `scheduled_release { apply_at_sample, off_sample }` を保持し、元のenvelopeも保持する。
現在hop発行のreceipt、実recipeと一致するonset、現在のalive/open policy、onset以降の予約を要求する。
発行時刻を起点に、予約Offを含むhopの開始を適用時刻とする。適用前は元のenvelope、適用以後は
予約で短縮したenvelopeを使用する。自然終端を予約で延ばすことはない。後続controlやhop不連続が
ない場合の条件付き予測であり、その後に実現した指令から発行済み予測を書き直さない。

この区別はattack中の短いreleaseで必要になる。現rendererはOff受領時にattack長をhold長で
切り詰めるため、予約を最初からToneへ適用すると、Off到着前の実音まで変わる。
今回はlive Toneを変更せず、予測の固定priorでhop前後を分けた。終端特徴も既知の短い終端を使う。
モデル名を `source_energy_log1p_residual_v2` に変更し、元のv1登録・source hash・検査結果は
`i10-source-energy-model.json` の `prior_registration` に保持した。残差学習の式、対照のマスク、
学習前採点、20 msのonset窓と16分割は維持した。descriptor教師は従来どおり介在指令を除外し、
予約Offを新しく学習許可する変更は行っていない。

通常renderer・capture・predictorを通す回帰検査は36条件を通過した。onsetと同時のOff、hop末尾、
hop境界、次hopのattack中、窓末尾付近、予約前の自然終端の6時刻条件を、正常／receiptなし／
発行時刻不一致／onset不一致／onset前release／閉じたpolicyと組み合わせた。
両busの実音はreceiptを外した対照と一致した。独立した区分式による固定prior、適用前後の
live envelope、action-independent priorも検査した。

既存6素材を `target/i10-scheduled-release-20260918/` へ再取得した。
保存入力列、172 PCMファイル、動作観測とsource-energy教師は元のreceipt corpusと一致した。
受動観測はbus／source epoch／frame IDで対応させ、共通503 frameの内容が一致した。
非同期の最新frame取得による片側121件ずつの非対応とdelivery latencyは、同一性を主張しない。
予約39件を通常reportで照合した。これらの予約はonsetの20 ms窓より後であり、固定priorの
1,280座標は変わらなかった。既存素材だけで変更動作を証明したとはしない。

短いrelease用の新development素材も取得した。最初の `short/` は `cycles` が整数専用のため
コンパイルで失敗した。`short-v2/` は取得に成功したが、実際のholdが0.25秒で窓外だった。
両結果を残し、`short-v3/registration.json` でflowのadaptive duration範囲を0.01に固定した。
結果は `target/i10-scheduled-release-short-v3-20260918/` にある。13件の予約を照合し、
両busの固定prior 416座標中398座標が元の予約なし式から変化した。保存recipe／envelope／
実時刻からの独立再計算との差は全座標で0だった。固定priorの実教師に対する二乗誤差総和は
0.0013602048530891495から0.0002604812213966682になったが、単一development sine素材の
診断であり、held-out精度、全身体への転用、学習済みモデルの採用を示さない。
元のreport中7 onset、新素材2 onsetは保存入力traceの終端後であり、recipe数値照合から除外した。

独立検査器は `scripts/verify_temporal_scheduled_release.py`。
適用hop、予約欠落、Off時刻、base envelopeの先行短縮、固定energy、attack長の6変異を拒否した。
元corpusは変更せず、一時コピーで変異検査した。再現例:

```bash
CONCHORDAL_I10_DEFAULT_INPUTS=target/i10-scheduled-release-inputs-20260918 \
CONCHORDAL_I10_DEFAULT_OUTPUT=<fresh-directory> \
RUST_BACKTRACE=1 cargo test --lib \
  runtime::body_profiles::policy_default::acquire_policy_defaults -- --ignored --exact --nocapture
python3 scripts/verify_temporal_scheduled_release.py <fresh-directory> \
  target/i10-scheduled-release-inputs-20260918 --previous target/i10-onset-receipts-20260918
```

短い素材では入力を同ディレクトリ内 `short-v3` に替え、検査器の `--previous` を省く。
通常全体テストは1,147成功、0失敗、35 ignored、31 suite。
`cargo test exit=0 @ 2026-09-18T09:58:38+09:00`。通常Clippy、全target check、fmt、diffも通過した。
必須テストログ、source hash、取得・検査結果は入力artifactへ保存した。

I10全体は未完。次は通常の共有候補モデルにおける実recipe・routing・envelope・周期・前史の
条件付けと同一機会のdefault対応である。今回の発行済みonset予測を、候補7 classの許可や
実身体適合へ昇格しない。共同状態・path寿命・条件付きscore、全head、評価時刻の判断、
全I10資源検査も引き続き残る。I9保留とI11作用無効を維持する。

## routingと身体応答の分離比較、および既存音の混入を特定（2026-09-18）

前節は進展として維持する。今回の対象はI10の共有profileから実身体への転用である。
既存監査で見つかった「routed実身体がunrouted medoid busへ対応する24件」について、
同じ身体・同じ発行時刻・同じ動作分岐のrouted busを使う比較版を作り、元版との誤差と順位差を
調べることを終了点にした。medoidの差し替え、閾値緩和、失敗身体の除外は行わない。
登録は `target/i10-routing-projection-20260918/preflight.json`、結果取得前の具体的な入力hashは
同ディレクトリの `registration.json` に保存した。

コンパイラへ明示的な `--routed-body` 比較モードを追加した。元のdescriptor source busがroutedなら
そのbusを使い、unroutedなら同じ取得済み身体の最小番号のrouted busを使う。routed素材がなければ
失敗させる。descriptorの出所busは別フィールドに保持する。PAMの8 medoid、六記述値、尺度、
0.25の距離閾値、tie、32時刻grid、全recipeは維持した。新profileのSHAは射影方針の比較版を識別し、
元のbody-model hashは変更していないPAM／尺度を識別する。O14の適合・採用版へ昇格しない。
実身体の対象busがunroutedならenergyは厳密な0、log RMSは規定floorとし、他の生特徴は未知とする。
未来のspectral warmupや観測支持を、routingの代数だけから作らない。

新binaryは `routed-body.bin`、SHA256は
`0bcc466cbd77c9594cef65b94545ccaabc6009c2632fbd74476088ea36f418b2`。
15,898,830 bytes、441固有trajectoryである。8 medoid中、
`01-sine-110-0.04-habitat-bus1-cut009600` の129 bindingだけが、同じ身体のbus 1からbus 0へ変わる。
元モードの再コンパイルbinaryは従来のSHA
`033905eadd9a7c681a8568ea347c25d1c1482d49d708bc849b00607a85d51fb5` と一致した。
独立読戻しで1,032 binding、387,000 frame、4,257,000座標のf64／maskが取得済みの対象busと一致した。

同じ36 development素材・396分岐・両busの転用比較では、72 ownerのdescriptorと選択medoidが全て
元版と一致した。unrouted素材への誤対応は24件から0件になったが、転用精度は改善しなかった。
297,000 energy座標のRMSEは0.0006611760297442038から0.000760846056483151へ増えた。
各窓4,104候補対の順位反転は次のとおりである。

| 発行からの積分窓 | 元版 | routing分離版 |
| --- | ---: | ---: |
| 0.25秒 | 280 | 464 |
| 1秒 | 356 | 608 |
| 2秒 | 304 | 536 |
| 4秒 | 208 | 392 |

新しく増えた反転は全て初期無音の層だった。既存2 toneの層は四窓とも元版と同じ。
無音層では、実音で差がない候補間へ264／288／288／288件の差を新たに作った。
具体例 `sine-both-silent-gate` は発行時のactive tone数が0であり、skip後の実energyも0だが、
選ばれたprototypeの既存音が残り、1秒積分energyを0.0008083112955623099と予測した。
`silent-control-witness.json` に実metadata、assignment、分岐値を保存した。
この例のskipを元の実policy defaultと呼ばない。元素材の対照は無追加指令である。

原因の分離として、routing単独補正は不十分と判定する。身体の新しい発音を転用する際に、
prototypeの発行前から残る音まで持ち込んでいる。次は実身体の既存状態の継続と、通常一回の
新しいrecipeによる励起を分離し、実際のactive inventory・予約release・envelope・前史に
条件付ける必要がある。単に小さい値へ閾値処理したり、無音層を検証対象から外したりしない。

検証器は別のscalar積分でrendererのenergy metadataから実値を再計算し、binary frameから
予測値を再計算した。72 assignment、792 branch/bus、288順位集合を照合した。
energy、選択cell、assignment、unrouted busの偽energyの4変異も拒否した。
コンパイラの読戻し検査と合わせ、計算手続きの誤りとモデル転用の失敗を区別した。

通常loaderは `routing_projection` を持つ比較artifactを明示的に拒否する。現在の通常表は
実Voiceのrouting消費者を持たず、この比較版をそのまま読み込むと条件を失うためである。
既存の登録済みprofile、通常表の演算、実音には変更を加えていない。loader回帰検査へこの条件を追加した。

再現例:

```bash
python3 scripts/compile_temporal_action_profiles.py \
  --actions target/i10-medoid-actions-frozen-20260918 \
  --features target/i10-medoid-features-frozen-20260918 \
  --output <fresh-binary> --routed-body
python3 scripts/verify_temporal_compiled_profiles.py <fresh-binary> \
  target/i10-medoid-features-frozen-20260918 --output <fresh-verification-json>
python3 scripts/evaluate_temporal_profile_transfer.py \
  target/i10-routing-projection-20260918/registration.json <fresh-result-json> \
  --profiles target/i10-routing-projection-20260918/routed-body.bin
python3 target/i10-routing-projection-20260918/verify.py
```

通常全体テスト1,147成功、0失敗、35 ignored、31 suite。
`cargo test exit=0 @ 2026-09-18T10:08:55+09:00`。既存Python 2検査、Clippy、全target checkも通過した。
必須ログ・status、source hash、失敗を含むモデル比較結果を同じartifactに保存した。

I10全体は未完。routingだけの補正は採用しない。実身体条件付きの帰結モデル、通常の候補許可と
default対応、共同状態・path寿命・score、全head、評価時刻、全資源検査は残る。
既存状態と新規励起の分離を次の実装単位とする。I9保留とI11作用無効を維持する。

## 実際の残存toneと指令toneを通常source-energy予測で分離（2026-09-18）

前節は進展として維持する。今回の対象はI10の実身体条件付けであり、まず通常の指令予測で、
実際に残る身体と新しい一回の発音を分離した。従来はsource全体の実音energyを教師にする一方、
固定priorが指令対象の1 toneだけだった。終了点は、通常rendererの実toneから残存成分を組み立て、
指令成分と別々に記録し、入力・計算・実音不変を検査することである。
実装前条件は `target/i10-retained-energy-inputs-20260918/preflight.json` に固定した。

`source_energy_log1p_residual_v3` は、受理した指令を適用した時点の同一source／source generationの
renderer entriesから、指令対象toneを除く成分を読む。各tone自身のrouting、現在amplitude、
envelope、発行時に既知の予約Offを保持する。前節の予約OffはRoutedToneにも保持し、後の指令の
予測で残存toneを読む際にも使う。実ToneのOff適用時刻は変更しない。
新規OnだけでなくOff予測も同じ経路を通る。別toneがpresentationだけに鳴っていれば、指令toneを
releaseした後もpresentationのsource-energy priorに残る。別sourceや別source generationは混ぜない。

固定priorは各成分の `amplitude^2 * envelope_gain^2 / 2` の和である。指令成分を
`command_fixed_energy`、残存成分を `retained.fixed_energy` として報告する。
位相を持った音圧の和ではなく、位相間の干渉を無視したsine-envelope近似である。
backend、modulator、未来のparameter update、後続指令は再現していない。同じbatch内でも、
後に適用する指令を先に取り込まない。発行済み予測は後続指令によって書き直さない。
body-generation不一致などの実教師の支持判定は、既存private captureの規則を維持する。

走査はsource IDで限定したBTree範囲に対し64 entryとoverflow判定1件までで打ち切る。
指令toneや別generationとして除外するentryも走査数に含める。65件目があれば
`retained.complete=false` とし、全sourceの三比較予測を未知にする。部分和を全体の予測として
採用せず、そのforecastから残差の採点・学習もしない。指令成分だけは別欄に残す。
固定配列を使い、新しいheap割当やDSP renderは追加しない。この局所上限は全I10資源検査の合格ではない。
`included_tones` は保持entry数であり、queued toneやtailも含む。可聴tone数とは呼ばない。

通常renderer・capture・predictorの回帰検査では、habitatのみ／presentationのみの残存tone、
別source、別generation、予約release、自然終了後の除去、新規OnとOffを組み合わせた。
両busの実音が観測なしrendererと一致し、残存成分が独立した区分式と一致した。
指令toneのrelease後にpresentation側の別toneのenergyが残る例も検査した。
通常observerを通した64／65 entry境界では、overflow時の予測が未知であり、有限の教師を与えても
部分priorを学習しないことを検査した。

既存6素材の再取得は `target/i10-retained-energy-20260918/`。
元の予約release版と入力列、172 PCMファイル、指令観測、source-energy教師が一致した。
受動観測は同一bus／epoch／frame IDの438件が一致した。片側186件ずつの非対応とdelivery latencyは
非同期reportの差として残し、全行の一致は主張しない。

`scripts/verify_temporal_retained_energy.py` は保存した指令列を順に読み、tone生成、Offによる
短縮、hop開始時の期限切れ、所有世代と各routingを独立に再構成する。
保存trace範囲の70 forecast・両bus・2,240窓で指令／残存／合計成分を照合し、最大絶対差は0だった。
残存成分が正の窓は1,199、従来の固定priorから値が変わった窓は1,197だった。
残存数、走査数、complete flag、残存energy、指令energy、合計energyの6変異を拒否した。
指令単独の固定priorとaction-independent予測は元版と一致した。

ただし、2,220の支持された比較窓で予測誤差は改善しなかった。実教師に対する二乗誤差総和は:

| 比較 | 元の指令単独版 | 残存成分を加えた版 |
| --- | ---: | ---: |
| 固定prior | 0.0077550229189244785 | 0.013481236695798695 |
| action-conditioned残差 | 0.008543051631873381 | 0.010431522993000845 |
| action-independent対照 | 0.01270896428553889 | 0.01270896428553889 |

harmonic素材では固定誤差が減ったが、sine-flowとmodal-flowでは増えた。この結果は実状態入力の
接続を証明する一方、この素材ではamplitude／envelopeの加算だけで精度が改善しないと確認した。
通常の研究診断として成分と誤差を残す。音響精度の改善、候補順位、ordinal校正、作者採用へは
昇格しない。過去版の登録と検査結果は `i10-source-energy-model.json` の `prior_registrations` に保持した。

再現例:

```bash
CONCHORDAL_I10_DEFAULT_INPUTS=target/i10-retained-energy-inputs-20260918 \
CONCHORDAL_I10_DEFAULT_OUTPUT=<fresh-directory> \
RUST_BACKTRACE=1 cargo test --lib \
  runtime::body_profiles::policy_default::acquire_policy_defaults -- --ignored --exact --nocapture
python3 scripts/verify_temporal_retained_energy.py <fresh-directory> \
  target/i10-retained-energy-inputs-20260918 target/i10-scheduled-release-20260918
```

全体テスト1,149成功、0失敗、35 ignored、31 suite。
`cargo test exit=0 @ 2026-09-18T10:21:25+09:00`。Clippy、全target check、fmt、diffも通過した。
必須ログ・status、検査対象source hash、取得と独立照合の結果を入力artifactへ保存した。

I10全体は未完。次はbackend／modulator／位相による実応答を含む身体条件付けと、候補7 class・
実defaultへの接続である。この指令時の加算近似で共有profileの転用問題が解決したとはしない。
共同状態・path寿命・条件付きscore、全head、評価時刻、全資源検査も残る。
I9保留とI11作用無効を維持する。

### 実modulatorの発行時状態を自声energy予測へ接続（2026-09-18）

前単位の加算priorは外側のADSRだけを使い、実際のEntrainPulseによる二段目の振幅減衰を
落としていた。`source_energy_log1p_residual_v4` は指令toneと残存toneの両方について、
発行時のmodulator状態、予定kick、既知の励起開始を使う。EntrainPulseのstate／level、
SeqGateのtimer、DroneSwayのphaseと発行時alphaを定数時間で外挿し、既存の16窓の
sine-envelope energyへ振幅の二乗を掛ける。指令のsnapshotは通常reportへ保存する。
実音rendererの状態を進めず、backendの複製・追加render・hop内の新規heap確保は行わない。

これはsampleごとのf32演算の閉形式近似であり、厳密な音響再生ではない。予定kickの後に
正のsustainからIdleへ戻らないEntrainPulseは支持できる。一方、未知の自律再発音が結果へ
影響する状態、非有限パラメータ、反復加算の丸めで終了判定が揺れるSeqGate境界は未知とする。
未励起toneは既知の励起までゼロ。外側envelopeがゼロなら内部modulatorの未知性にかかわらず
ゼロを保持する。`retained.complete` は走査の完全性だけを表し、追加の
`retained.control_supported[16]` が窓ごとのcontrol支持を表す。支持を欠く合計から誤差採点や
残差学習を行わない。各toneのrouting、予約Off、世代、64件とoverflow probeの制限は維持する。

実装前に `target/i10-control-amplitude-inputs-20260918/preflight.json` を固定した。
実modulatorをsampleごとに進める比較では、8／48 kHz、複数の発行時状態、即時／遅延kick、
三種類のmodulatorの432点を照合した。最大振幅差は0.001179196276783645。予測を発行しても
元のmodulator状態は変わらない。自律状態の未知、未励起、kick後の支持、SeqGate境界、
未知成分から学習しない回帰も検査した。

元の6 policy素材を `target/i10-control-amplitude-20260918/` へ再取得した。
独立Python照合は保存された実指令からEntrainPulseのf32状態を再構成し、70予測・2,240窓の
指令／残存／合計energyを照合した。計算差はゼロ。172音声ファイル、実指令と教師、共通の
429観測frameは不変。非同期の未対応frameは同一性検査へ含めない。誤った開始時刻、state、
支持maskを含む9変異はすべて拒否した。旧v3照合経路も既存素材で再検査した。

支持された2,220窓の二乗誤差合計は次のとおり。これはdevelopment素材上の診断である。

| 比較 | v3 | v4 |
| --- | ---: | ---: |
| 固定prior | 0.013481236695798695 | 0.007761865430497955 |
| action-conditioned残差 | 0.010431522993000845 | 0.007877097873786114 |
| action-independent対照 | 0.01270896428553889 | 0.01270896428553889 |

合計は改善したが、固定priorはsine-hold、harmonic-entrained、harmonic-pulseで悪化した。
backend固有の応答、sineのimpulse boost、位相干渉、未来のparameter／rhythm更新は未対応。
この結果から一般的な身体転用精度、候補順位、校正、可聴性、作者採用を主張しない。

再現例:

```bash
CONCHORDAL_I10_DEFAULT_INPUTS=target/i10-control-amplitude-inputs-20260918 \
CONCHORDAL_I10_DEFAULT_OUTPUT=<fresh-directory> \
RUST_BACKTRACE=1 cargo test --lib \
  runtime::body_profiles::policy_default::acquire_policy_defaults -- --ignored --exact --nocapture
python3 scripts/verify_temporal_retained_energy.py <fresh-directory> \
  target/i10-control-amplitude-inputs-20260918 target/i10-retained-energy-20260918 --controls
```

全体テスト1,152成功、0失敗、35 ignored、31 suite。
`cargo test exit=0 @ 2026-09-18T10:45:06+09:00`。Clippy、全target check、fmtも通過した。
対象source hash、全体testログとstatus、取得・独立照合・変異検査の結果を入力artifactへ保存した。

I10全体は未完。次は既存の実身体状態と新規励起の応答を分けた候補7 class／defaultへの
投影接続であり、今回の指令窓モデルをその代わりにはしない。backend・位相の条件付け、
共同状態とpath寿命、条件付きscore、全head、評価時刻、4／16／64 Voice・両busの
全資源検査も残る。I9保留とI11作用無効を維持する。

### 通常の身体energy核を7 class／local default比較で共用（2026-09-18）

対象はI10の実身体条件付けである。通常の指令／残存tone予測を `ToneEnergy::at` へまとめ、
既知の予約Offと候補releaseをそれぞれの到着hop以降に適用する同じ核を、実機会の7 classにも
通した。早いreleaseはattackを到着前に切り詰めず、遅い追加releaseは元の終端を延長しない。
通常経路はこの核を消費する。7 classの列挙とdefault比較はまだofflineであり、通常の
候補許可・決定schedule・共有帰結表への投影接続を完了したとはしない。

`target/i10-seven-class-energy-inputs-20260918/registration.json` で入力と実音targetを固定した。
予測器は実policyの最初のgrantまでを逐次読み、その後の指令・rhythmもtarget音声も読まない。
実rendererの残存tone状態、各routing、予約release、同じ一回のrecipeから予測する。
onset／delayだけが新規励起を追加し、wait／skip／continueは既存の自然経過を保持する。
release／gapは候補時刻に活動中の既存toneだけをreleaseする。これらの機会bookkeepingは
保存するが、liveの時計や行動を更新しない。defaultは当該grantのonsetと既知releaseであり、
後続の実policy指令列全体とは区別する。

`target/i10-seven-class-energy-20260918/` に4素材・71分岐を取得した。Hold／silentの2素材は
記録区間にgrantがないため、その事実を明示し候補を捏造しない。両busと0.25／1／2／4秒の
物理窓を各16点で予測した。独立Python参照は実指令から残存inventoryとmodulator状態を
再構成し、candidate入力、shiftしたrecipe、予約と追加Off、9,088点、568窓の平均とdefault差を
照合した。計算差はゼロ。bus／窓／点の欠落を含む14変異はすべて拒否した。

実音PCMを独立積分した窓平均との比較は次のとおり。各列は同じ71分岐・両bus・4窓を使う。
「内側controlなし」はmodulatorと発音開始の係数だけを外し、「残存toneなし」は新規励起だけを
残した比較である。研究用の固定energy比較であり、ordinal評価や望ましい行動の順位ではない。

| 比較 | 実状態controlあり | 内側controlなし | 残存toneなし |
| --- | ---: | ---: | ---: |
| 窓平均の二乗誤差合計 | 0.0000225432081856134 | 0.0002150777677528619 | 0.000028277526598169835 |
| local default差の二乗誤差合計 | 0.00009735058720189645 | 0.0001224562109956027 | 0.00010569954051779009 |
| energy順位の反転 | 54 / 4,936 | 57 / 4,936 | 54 / 4,936 |
| 実音の差を予測が同順位とした対 | 49 / 4,936 | 44 / 4,936 | 1,114 / 4,936 |
| 実音が同順位なのに差を作った対 | 0 / 4,936 | 0 / 4,936 | 0 / 4,936 |

順位は平均energy差の符号をそのまま比較し、結果を見てepsilonを調整しない。素材別・窓別・
bus別の結果は `verification.json` に保持した。合計だけで成功としない。sine-flowでは内側
controlなしの方がdefault差の誤差が小さく、modal-flowでは残存toneなしの方が窓平均とdefault差の
誤差が小さい。harmonic-pulseでもcontrolなしの誤差が小さい。

実音上で、加算近似そのものが表せない反例も得た。sine-flowの24,000 sample遅延onsetを1秒窓で
評価すると、両busで残存音のenergyは0.00044070953354162983。保存済みの混合音から残存音を
引いた残差のenergyは0.00027023605932087626、交差項は-0.0003651894811999977となり、
onsetによる実energy差は-0.00009495342187912137に下がる。一方、予測は
+0.0002608516941737726と増加させた。非負energyの加算ではこの相殺を表せない。
これは160窓の探索的分解中の2窓であり、残差にはf32混合丸めも含む。独立した新tone録音と
偽らず、`interference-diagnostic.json` にその定義と全結果を残した。閾値やfitは変更していない。

共用化の通常経路回帰は `target/i10-seven-class-energy-ordinary-20260918/` に再取得した。
97件の指令予測・教師を含むoutcome全体、実policy入力、172音声ファイル、共通483観測frameが
旧v4と一致した。非同期未対応frameは同一性を主張しない。元のv4近似を変更した版ではなく、
同じ数値核で候補とdefaultを比較できるようにした版である。

再現例:

```bash
CONCHORDAL_I10_ENERGY_INPUTS=target/i10-seven-class-energy-inputs-20260918 \
CONCHORDAL_I10_ENERGY_OUTPUT=<fresh-directory> \
RUST_BACKTRACE=1 cargo test --lib \
  runtime::body_profiles::policy_default::energy_projection::acquire_seven_class_energy \
  -- --ignored --exact --nocapture
python3 scripts/verify_temporal_candidate_energy.py <fresh-directory> \
  target/i10-seven-class-energy-inputs-20260918
```

全体テスト1,153成功、0失敗、36 ignored、31 suite。
`cargo test exit=0 @ 2026-09-18T11:01:14+09:00`。Clippy、全target check、fmt、Python構文検査も通過した。
対象source hashと全体testログ・status、取得・照合・変異検査の結果を入力artifactへ保存した。

I10全体は未完。次の身体予測では、残存音と追加励起の位相を含む応答を扱い、この反例と
候補順位を再検査する。通常の候補／実default消費と共有profileの身体条件付け、共同状態・
path寿命・条件付きscore、全head、評価時刻、全資源検査は引き続き残る。
I9保留とI11作用無効を維持する。

### 実sine位相と搬送波の交差項を通常予測へ接続（2026-09-18）

前単位で見つかった相殺の反例に対し、実oscillatorの発行時x／yと回転係数、未発音toneの
初期seedを読み取り、`source_energy_log1p_residual_v5` を通常指令／残存toneと7 class比較で
共用した。Toneやbackendを複製・進行させず、静止したpitch／amplitudeのsineについて、
位相と回転normを外挿する。sine impulse boostとその減衰も含める。各16分割窓では外側
ADSR・内側modulatorとboostの振幅をmidpointで固定し、搬送波どうしの積を整数sample区間で
解析積分する。既知のonset／release端で各自己項と交差項の区間を切り詰める。

遅延onsetは元のseedを単純移動せず、実rendererと同じsource／onset／tone IDからseedを作る。
seedの読み取りは乱数状態を消費しない。すべての非ゼロ寄与が対応範囲にある窓だけでcoherent値を
使う。非sine、未消化parameter更新、平滑化中のpitch／amplitude、未対応controlや容量超過は
`coherent_sine_energy=None` とし、従来の近似を `incoherent_fixed_energy` に明示して保持する。
このfallbackを位相対応の成功へ数えない。元の支持欠落やinventory切り捨てから学習しない規則も
維持する。private reportへ保存する位相は聴取観測や共有memoryへ投入しない。

入力と旧実音targetは `target/i10-coherent-sine-inputs-20260918/registration.json` に固定した。
実Toneをsampleごとに進める84点比較で、予測sampleの最大絶対差は0.0000023388246175592986。
混合波の直接加算・二乗平均との90条件比較は、同周波数、近接周波数、異周波数、逆位相、
部分重複、巨大な絶対sample時刻を含み、1e-12以内で一致した。未知と64／65成分境界も検査した。

4素材・71分岐は `target/i10-coherent-sine-20260918/` に取得した。Python側で初期seedと
残存sineのf32回転・boostを逐次再構成し、別の直接波形和でcarrier-product平均を照合した。
9,088点の最大計算差は2.6020852139652106e-18。位相、boost、coherent支持、fallback値、
bus／窓／点の欠落を含む19変異を拒否した。旧v4の比較結果も同一入力で再計算し、一致を確認した。

sine-flowの1,520候補対では、実音energy順位の反転が30→0、実差を同順位とした対が8→0となった。
新しく差を捏造した対は0。全4,936対では反転54→24、同順位への潰れ49→41であり、
harmonic／modalの値と失敗は旧近似のまま残る。

| 71分岐・両bus・4物理窓の比較 | v4 | v5 |
| --- | ---: | ---: |
| 窓平均の二乗誤差合計 | 0.0000225432081856134 | 0.0000177009275994668 |
| local default差の二乗誤差合計 | 0.00009735058720189645 | 0.000013638034270658828 |

既知の反例であるsine-flowの24,000 sample遅延onset、1秒窓では、無追加励起に対する実energy差
-0.00009495342187912137に対し、v4は+0.0002608516941737726、v5は
-0.0000954219386000488を予測した。これはこの固定development素材の符号・数値・順位の改善であり、
一般的な身体転用、ordinal帰結、校正、可聴性や作者採用の証拠とは区別する。

通常の6素材も `target/i10-coherent-sine-ordinary-20260918/` に再取得した。70予測・2,240窓の
従来成分とcoherent選択を確認し、172音声ファイル・実指令・教師・共通478観測frameは不変。
支持された2,220窓の二乗誤差合計は、固定予測0.007761865430497955→0.006563818141556952、
action-conditioned残差0.007877097873786114→0.0068460453842649795。
action-independent対照0.01270896428553889は不変。harmonic／modalの誤差も不変である。
通常成分の検査器はcarrier積を再証明せず、その検証は上記のseed再生・波形和比較と分けて記録した。

計算は64成分に制限し、追加のheap確保やDSP renderはない。両bus各16窓の一時scratchは
82,432 bytes。一般の異周波数で交差項は二次の計算量となる。test profileの核単独257反復では、
1／4／16／64 toneのp99が2,010／13,000／180,981／3,018,580 nsだった。この測定はcontrolや
observer全体、同時実音処理を含まない。実時間の受入は未達として保持する。release profile（opt-level=3／LTO）でも64 toneの核単独p99は
2,567,746 nsであり、追加処理枠1.6 msを超えた。低負荷での改善を全負荷の合格へ昇格せず、
計算費用の削減と共有budgetを次の実装課題に残す。

再現例:

```bash
CONCHORDAL_I10_ENERGY_INPUTS=target/i10-coherent-sine-inputs-20260918 \
CONCHORDAL_I10_ENERGY_OUTPUT=<fresh-directory> \
RUST_BACKTRACE=1 cargo test --lib \
  runtime::body_profiles::policy_default::energy_projection::acquire_seven_class_energy \
  -- --ignored --exact --nocapture
python3 scripts/verify_temporal_candidate_energy.py <fresh-directory> \
  target/i10-coherent-sine-inputs-20260918
cargo test --release --lib \
  life::self_prediction::projection::coherent_tests::coherent_window_cost_probe \
  -- --ignored --exact --nocapture
```

全体テスト1,155成功、0失敗、37 ignored、31 suite。
`cargo test exit=0 @ 2026-09-18T11:23:05+09:00`。Clippy、全target check、fmt、Python構文検査も通過した。
最初の全体検査では旧固定priorをそのまま学習初期値とするtestが失敗し、初期残差ゼロを新しい
固定priorに対して検査するよう更新した。最初の失敗log／statusも入力artifactへ保持した。

I10全体は未完。位相計算の費用と残るbackend応答を扱い、候補順位・default差を再検査する。
通常の候補／実default消費と共有profileの実身体条件付け、共同状態・path寿命・条件付きscore、
全head、評価時刻、全資源検査は残る。I9保留とI11作用無効を維持する。

### 位相積分の区間端を再利用し、計算費用を削減（2026-09-18）

v5の予測対象・支持・midpoint振幅近似を変えず、整数sampleのcarrier積分を区間端の位相で
計算する形へ整理した。各toneで半sampleずらした両端の位相と半角回転、midpointからの
時刻差を一度だけ計算し、共通端を持つ交差項で再利用する。発音・release端が異なる組は
重なり区間の端で必要な回転だけを追加する。非特異な交差項では和周波数と差周波数の項を
一つの除算へまとめた。候補の削除、周波数差による交差項の省略、容量の縮小は行っていない。

用いた離散積分の恒等式は、整数sample区間 `[a,b)`、整数midpoint `m` に対して

```text
sum cos(omega*(k-m)+phi)
  = [sin(omega*(b-m-1/2)+phi) - sin(omega*(a-m-1/2)+phi)]
    / [2*sin(omega/2)]
```

となる。分母が小さい自己項・交差項では中心位相による元の式へ戻し、同周波数差ゼロは
定数として直接積分する。追加検査でNyquist近傍の旧式の桁落ちも検出したため、この経路では
周波数を整数sampleのaliasとして `[-pi,pi]` へ戻してから比を計算する。この境界検査は
積分核の検査であり、通常sine backendの0.49 fsを超える成分のcullを変更するものではない。

既存90条件に加え、1／4／16／64 tone、1／7／4096 sample、同一・近接・異周波数・ゼロ付近・
Nyquist付近、異なる区間端、巨大な絶対時刻を組み合わせた84条件を直接波形和で検査した。
最終実装の最大絶対差は1.7335833696298028e-15。最初の境界検査は失敗を検出し、上記alias処理を
加えた後に通過した。上限64と未知／65成分の拒否は維持する。

入力・検査logは `target/i10-coherent-edges-inputs-20260918/`、最終候補出力は
`target/i10-coherent-edges-checked-20260918/`、通常出力は
`target/i10-coherent-edges-ordinary-checked-20260918/` に保存した。7 class・71分岐の
9,088点を既存の独立波形和と照合し、最大差2.6020852139652106e-18。
最適化前との27,231浮動小数値の比較では404値が変わり、最大差2.168404344971009e-19。
全4,936候補対の順位判定は不変で、sine-flowの反転ゼロを維持する。

通常70予測・2,240座標も再取得・独立照合した。固定予測244値に丸め差があるが、
2,220支持窓の固定／条件付き／blindの誤差合計は保存されたf64値で不変。
172音声ファイル、実指令と教師、共通485観測frameは不変。共通frame数はこの取得対の
照合範囲であり、非同期出力の全frame同一を意味しない。

両bus各16窓の入力scratch 82,432 bytesに、積分呼出しで共用する64組の端点cache
4,096 bytesを加え、計86,528 bytesの明示的な配列領域を使う。追加heap確保はない。
計算量は引き続き二次であり、全observerや実音処理の費用をこの核の測定で代用しない。

最初のrelease測定では、64 tone・32窓の同一区間p99が2,567,746→367,703 nsへ下がったが、
異なる端点の条件は1,664,240 nsで1.6 ms枠を超えた。この失敗を
`first-release-cost.json` とそのsource snapshotに保持した。その後、除算の整理と端点時刻差の
cacheを追加した。

続いて異なる端点と近接周波数を組み合わせると、同じ核を切り出したstandalone probe
（rustc opt-level=3／LTO）でp99 2,505,516 nsを観測した。単独条件の合格だけでは不十分と
確認し、複合条件も通常のignored費用probeへ追加した。小角の中心位相とsinc比について、
引数の絶対値が0.01以下の範囲だけで6次多項式を使う。各打切り剰余は3e-21未満であり、
ここでのf64丸め精度より小さい。範囲外は元の三角関数へ戻す。交差項そのものは省略しない。
直接波形和の84条件は、この小角域と範囲外も含む。最終release probeは6支持・周波数条件と
1／4／16／64 toneの全24条件を測定した。

| 最終release、64 tone・32窓・257反復 | median ns | p99 ns |
| --- | ---: | ---: |
| 同一区間・異周波数 | 262,421 | 270,341 |
| 異なる端点・異周波数 | 1,245,528 | 1,289,848 |
| 同一区間・近接周波数 | 703,854 | 731,495 |
| 同一区間・同周波数 | 317,762 | 327,512 |
| 異なる端点・近接周波数 | 1,596,389 | 1,620,910 |
| 異なる端点・同周波数 | 1,297,648 | 1,314,758 |

1.6 ms枠を超える条件は残る。最終測定の超過を消さず、共有budgetと全負荷受入を未達として保持する。

周波数が既に `[-pi,pi]` にある場合の不要なalias計算も省いた。予測対象や支持は変更していない。

全体テスト1,156成功、0失敗、37 ignored、31 suite。`cargo test exit=0 @ 2026-09-18T11:55:14+09:00`。
Clippy、全target check、fmt、既存の独立候補／通常検査器も通過した。source hash、最終source、
全体test log／statusと各取得・照合結果を入力artifactへ保存した。

再現例:

```bash
CONCHORDAL_I10_ENERGY_INPUTS=target/i10-coherent-edges-inputs-20260918 \
CONCHORDAL_I10_ENERGY_OUTPUT=<fresh-directory> \
RUST_BACKTRACE=1 cargo test --lib \
  runtime::body_profiles::policy_default::energy_projection::acquire_seven_class_energy \
  -- --ignored --exact --nocapture
python3 scripts/verify_temporal_candidate_energy.py <fresh-directory> \
  target/i10-coherent-edges-inputs-20260918
cargo test --release --lib \
  life::self_prediction::projection::coherent_tests::coherent_window_cost_probe \
  -- --ignored --exact --nocapture
```


I10全体は未完。共有budgetと全Voice負荷の受入、非sineの身体応答、通常候補／実default消費と
共有profileの身体条件付け、共同状態・path寿命・条件付きscore、全headと評価時刻が残る。
I9保留とI11作用無効を維持する。

### 共有候補表の再構築周期と資源診断を通常workerへ接続（2026-09-18）

`action_profiles::Table` は通常のrelation workerで共有表を作っているが、これまで
prototype/group割当が変わると100 ms内でもcache missとして再構築していた。
割当の失効をすぐ反映しながら、再構築の時計は失効でリセットしない形へ修正した。
同じprofile SHA・bus・epochでは前回構築から4,800 sample以上経過した最初のcanonical hopで
再構築する。待機中は表を返さず、失効したgroupを含む古い混合を使わない。
モデル・epochが変われば古い表を再利用しない。profileと身体モデルの不一致、未来の割当支持、
同一時計の巻き戻りは拒否する。

上限は引き続き8 prototype×7 class×32 offset、1構築あたり最大1,792 projection呼出し。
これは共有workerの表構築の制限であり、Voice側の20 Hz、64 decision/hop、128 decision/100 msを
実装したという意味ではない。後者の通常消費・共有budgetは残る。
当面のat-action評価時刻、raw head、profileの転用未受入、I11作用無効も変更していない。

`temporal_observation.action_profile_resources` に構築・失効・延期・入力拒否・projection呼出しの
累積回数、最終構築sample、最終／最大／累積構築時間を追加した。profile機能を設定していない場合はこの新しい
欄をserializeしない。UIにも回数と最終／最大時間を表示する。時間は構築を実行したrefreshの
入口から出口までのwall timeであり、構築前の検査・失効処理を含む。cache hit、延期、入力拒否の
呼出しは回数だけを記録し、全worker CPUやpublication latencyの測定とは区別する。
既存の表領域を再利用し、周期管理やカウンタのためのheap確保は追加していない。

因果的な単体検査では、実際のcanonical-hop観測から表を作り、group割当の除去・復帰を挟んでも
周期が満ちるまでprojectionを再実行しないこと、同じissueのcache再読が不変であること、
未来支持・巻き戻りの拒否が周期をリセットしないことを確認した。観測側のPhrase状態は
refresh前後で不変。既存の失効検査も「空表の即再構築」から「周期待ちの表なし」へ更新した。

実音素材mixed／sustain／stairと設定、変更前binaryを
`target/i10-table-cycle-20260918/` に固定し、debugとreleaseの変更後binaryで再取得した。
共有profileは既存の `033905eadd9a7c681a8568ea347c25d1c1482d49d708bc849b00607a85d51fb5` を使った。
素材ごとの音声ファイルは変更前とbyte一致し、実指令・身体default・既存予測記録も一致した。
両busの終了状態を含む共通観測はdebug 270 frame、release 278 frameで一致した。
比較から除いた新しい資源欄は、別の独立検査で時計、単調カウンタ、構築ごとのセル上限、
失効groupと表の対応を照合した。

両profileで332構築、65,184 projection呼出し、周期待ち6回を記録した。
reportから独立に照合できた構築issueは276件であり、残る56件は間引かれたreportから個別の
構築時間を復元できない。reportに残る1,576 cellと3,152 raw head分布も既存の独立数値検査に
通した。上限超過、未来の構築時刻、カウンタ後退、時間集計の矛盾、古い表時刻の5変異を拒否した。

変更前reportで直接確認できた100 ms未満の構築間隔は0件だった。変更前コードの再構築分岐と
因果的fixtureで確認した欠陥を、間引かれた旧reportで実測済みとは扱わない。
新しい実音取得で延期された6回と、全構築回数を数える経路は今回から確認できる。

| 素材 | debug 最大構築wall time、habitat / presentation (ms) | release 最大構築wall time、habitat / presentation (ms) |
| --- | ---: | ---: |
| mixed | 122.668 / 122.429 | 71.426 / 71.361 |
| sustain | 190.810 / 190.379 | 113.859 / 114.655 |
| stair | 190.064 / 190.753 | 113.923 / 113.475 |

releaseでも1構築が100 ms周期を超える条件が残った。これは疎なprototype対応を含む3素材の
測定であり、全8 prototype対応や4／16／64 Voiceの受入ではない。間引きreportの時間値から
全構築のp99を作らず、別threadのこのwall timeをhop内decision枠の1.6 msとも混同しない。
再構築の重複は止めたが、1構築の重い計算を減らす課題が新たに具体化した。

再現例:

```bash
python3 target/i10-table-cycle-20260918/acquire.py after
python3 target/i10-table-cycle-20260918/verify.py
cargo build --release --bin conchordal-render
python3 target/i10-table-cycle-20260918/acquire.py release
python3 target/i10-table-cycle-20260918/verify_release.py
```

全体テスト1,157成功、0失敗、37 ignored、31 suite。
`cargo test exit=0 @ 2026-09-18T12:11:11+09:00`。Clippy、全target check、fmt、Python構文検査、
release buildも通過した。入力・source hash、取得command、検査結果、全体test log／statusを保存した。

I10全体は未完。次は共有表内の重複する特徴計算を再利用し、全worker・hop負荷を検査する。
通常候補／実default消費、共有profileの実身体条件付け、共同proposalとpath寿命、条件付きscore、
全headと評価時刻も引き続き残る。I9保留とI11作用無効を維持する。

### 共有候補表の同一全軌道を構築内で再利用（2026-09-18）

前節で確認した共有表の構築費用に対し、同じprototype・group・offsetの先行classが
同じ全軌道IDを参照する場合だけ、計算済みCellを再利用するようにした。軌道IDは読み込み時に
検査した不変profile内の375 frame全体を指す。`Profiles::frames` の変換は、この全軌道と
発行時刻・直前観測・background energyで決まり、class固有の別処理はない。
同じ構築内の同じprototypeなら、音群の前史、Gesture、arrival、凍結headも共通である。
再利用時にはclass metadataだけを対象classへ変更する。

異なる軌道ID、異なるoffset、異なるprototype、別のissueの結果は流用しない。
軌道prefixの近似一致や許容誤差による比較も使わない。欠落bindingや計算失敗は従来どおり
直接評価し、成功したCellだけを再利用する。raw head混合を含む数値、支持mask、未知質量、
出所、候補数、at-action評価時刻は変更していない。検索は先行class最大6件の有界走査であり、
既存Cell領域を使うため新しいheap確保や永続cacheはない。

資源診断には `projection_calls` と `reused_cells` を追加し、通常reportとUIに接続した。
各時点で `attempted_cells = projection_calls + reused_cells` を満たす。
`projection_calls` は欠落binding等でNoneを返す呼出しも含む。追加カウンタによるTableの
常駐領域増加は16 byteであり、既存の `table_bytes` に反映する。
旧記録の `attempted_cells` は当時すべて直接呼出しだったが、以後はこの二成分を区別する。

canonical-hopの観測と割当失効・復帰のfixtureを拡張し、同一軌道のclassと、energyおよび
支持maskが異なるrelease軌道を併置した。2回の構築の全448セル位置を、再利用しない直接計算の
serialize結果と完全一致で照合した。258有効セルのうち130を再利用し、318回は直接呼び出した。
残る190位置は欠落bindingである。観測状態の不変性、失効後の再構築、時計巻き戻りの拒否も維持した。

前節のrelease binaryと同じ入力・profileを固定し、変更後releaseでmixed／sustain／stairを
再取得した。成果物は `target/i10-table-reuse-20260918/` に保存した。変更前binaryのhashは
前節の取得command記録と一致する。今回も332構築、65,184セル位置、延期6回であり、
直接projection呼出しは53,635回、再利用は11,549回となった。

両取得に共通する275件の公開表は、Table領域の16 byte増加を除き、全snapshotの
JSON数値・metadata・支持が一致した。音声3ファイルもbyte一致し、終了両busを含む
281共通観測、実指令・身体default・既存予測記録も一致した。reportは間引かれるため、
これは全構築・全セルの直接比較ではない。sustain／stairの1,576公開セルと3,152 raw head分布は、
既存の独立数値検査にも通した。呼出しと再利用の計数矛盾、および候補class改変の変異を拒否した。

| 素材 | 変更前release最大構築wall time、habitat / presentation (ms) | 再利用後release最大構築wall time、habitat / presentation (ms) |
| --- | ---: | ---: |
| mixed | 71.426 / 71.361 | 50.511 / 52.393 |
| sustain | 113.859 / 114.655 | 81.035 / 81.301 |
| stair | 113.923 / 113.475 | 82.217 / 81.034 |

今回の3素材では最大構築時間が100 ms未満になった。ただし同じ条件を一回ずつ取得した
wall timeであり、汎用の速度倍率やp99の保証ではない。全8 prototype対応、4／16／64 Voice、
全workerのpublication遅延、decision合計1.6 msとhop合計8.5333 msの受入は未実施である。
これらを表単独の測定で合格に置き換えない。

再現例:

```bash
cargo build --release --bin conchordal-render
python3 target/i10-table-reuse-20260918/acquire.py
python3 target/i10-table-reuse-20260918/verify.py
```

全体テスト1,157成功、0失敗、37 ignored、31 suite。
`cargo test exit=0 @ 2026-09-18T12:28:02+09:00`。Clippy、全target check、fmt、release build、
Python構文検査も通過した。入力・変更source・実行binary・取得command・検査成果物のhashと、
test log／statusを登録した。

I10全体は未完。通常の候補／実default消費、共有profileの実身体条件付け、共同proposalと
path寿命、条件付きscore、全head、評価時刻、全負荷受入を残す。次の資源検査では表構築単独と
worker／hop全体を分け、共有予算と通常消費の接続も含めて測定する。I9保留とI11作用無効は維持する。

### 同一軌道のarticulation予測で完全hopの途中状態を再利用（2026-09-18）

共同正規化の通常接続を再点検したところ、共有contextの所有・寿命と完全な条件付き特徴layoutが
引き続き欠けていた。仮のcontext重みやscore差で接続を代用せず、独立に進められる資源課題を
先に扱った。§9.1の予算は、100 ms周期に蓄積する観測・matching・共有表を合わせて50 msである。
前節の表単独82.217 msはこの予算を満たさない。100 ms未満という前節の測定結果を、
全workerの受入に読み替えない。

固定profileの全生特徴prefixを比較すると、同一評価時刻でのCell再利用は、通常prototypeで
有効129セル中の直接94→92セル、無音prototypeでは32→32セルにしか減らない。
この追加indexは実装せず、同じ全軌道の異なる評価時刻に対して、articulationを毎回先頭から
進め直す重複を除いた。調査結果は成果物内の `prefix-census.json` に保存した。

`Gesture::project_states` に構築内のscratchを渡し、音群handle・発行sample・全軌道IDが同じで、
保存済み時刻が評価時刻以下の場合だけ、完全hop終端の15 path・各entered時刻・質量・累積枝刈り質量を
再利用する。評価窓の末尾が部分hopなら、その結果を途中状態へ保存しない。次の評価では
直前の完全hopからその端を計算し直すため、部分hopでの遷移・順位選択を後の窓へ持ち越さない。
評価時刻の巻き戻し、軌道切替、音群世代変更は先頭から計算する。

scratchは不変profileを使う一回のTable構築に閉じ、次の発行・別modelへ残さない。
heap確保は追加していない。cache payloadは448 byteであり、既存のpath展開scratchや
コンパイラが確保するstack全体の値とは区別する。直接計算の経路はcacheを持たない。
Tableの常駐領域には二つのu64計数だけを追加した（16 byte）。`articulation_frames` は実行した
遷移hop数（端の部分hopを含む）、`reused_articulation_frames` は省いた既存の完全hop数であり、
report/UIへ公開する。共有表のCell再利用で関数自体を呼ばなかった分は、従来の `reused_cells` で数える。

部分hop、同じ評価の反復、評価の巻き戻し、異なる軌道、音群世代変更を含む60条件で、
cacheなしの直接計算と全serialize結果が完全一致した。非ゼロのelapsed係数と既知unknown質量を
使い、実際に発生した枝刈り質量も照合した。さらにTableの2発行・全448セル位置を直接計算と
照合し、観測状態の不変性と、Cell再利用と途中状態再利用の併用を検査した。

同じmixed／sustain／stairとprofileをreleaseで取得し、前節のrelease取得と比較した。
共通283件の公開表はTable領域の16 byte増加を除いて全JSONが一致し、音声3ファイルはbyte一致した。
終了両busを含む290共通観測、実指令・身体default・既存予測記録も一致した。
sustain／stairの1,576公開セル・3,152 raw head分布は独立数値検査を通過した。
計数の分解矛盾、候補class改変、articulationの処理上限超過の3変異を拒否した。

全332構築のCell位置65,184、直接呼出し53,635、Cell再利用11,549、周期待ち6回は前節と同じ。
今回の途中状態再利用では3,249,553遷移hopを実行し、1,535,132完全hopの再計算を省いた。
この計数は実際にarticulation関数へ入った呼出しの範囲であり、Cell丸ごとの再利用を二重計上しない。
公開reportで構築ごとの時間が見えるのは290件であり、全332件のp99は算出していない。

| 素材 | 変更前release最大構築wall time、habitat / presentation (ms) | 途中状態再利用後release最大構築wall time、habitat / presentation (ms) |
| --- | ---: | ---: |
| mixed | 50.511 / 52.393 | 37.498 / 38.255 |
| sustain | 81.035 / 81.301 | 61.745 / 63.389 |
| stair | 82.217 / 81.034 | 61.892 / 62.046 |

構築費用は減ったが、今回も表単独のwall timeが50 msを超えた。全workerの予算達成を示さず、
4／16／64 Voice、全prototype対応、共同推論を含む処理の受入にも使わない。
実行時間は固定3素材を一回ずつ取得した値であり、速度倍率や全構築p99の保証ではない。

成果物と再現手順:

```bash
cargo build --release --bin conchordal-render
python3 target/i10-articulation-prefix-20260918/acquire.py
python3 target/i10-articulation-prefix-20260918/verify.py
```

全体テスト1,158成功、0失敗、37 ignored、31 suite。
`cargo test exit=0 @ 2026-09-18T12:43:11+09:00`。Clippy、全target check、fmt、release build、
Python構文検査も通過した。最終source、入力、前後binary、取得command、検査結果、全体test log／statusを
hash付きで登録した。初期のfocused logにある可視性警告は、Phraseの入口を実消費者のある
`temporal_cognition` 内へ絞って解消し、上記の最終全体検査で再検証した。

I10全体は未完。通常の共同推論には共有contextの所有・寿命、条件付き特徴layoutとscoreが必要である。
実身体へのprofile条件付け、通常候補／実default消費、全head・評価時刻、共有workerとhop全体の
資源受入も残る。I9保留、I11作用無効は維持する。

### 共有worker全体の処理・公開遅延と4／16／64 Voiceの現行負荷を計測（2026-09-18）

共有表単独の時間に加え、profileを有効にした通常relation workerの資源診断を
`worker_resources` としてreport/UIへ接続した。受信完了後から観測の検査・epoch再初期化・
全有効処理・reference公開・観測snapshotのlock待ちとcopyまでをwall timeで測る。
受信待機、計数器の更新と最後の資源field copy、外側のreport serialize、audio workerや
device callbackの処理は含めない。これはthread CPU時間ではない。
thread内の初期化、拒否入力、EOFのfinish処理は、それぞれ別の計数へ残す。
profileを設定していない経路には、この計数fieldとInstant計測を追加しない。

受理frameの開始sampleで、epochの開始からceil(sample_rate/10) sampleごとの非重複区間へ
処理費用を割り当てる。48 kHzでは4,800 sampleであり、区間あたり9または10 canonical hopとなる。
境界をまたぐhopの費用は、そのhopが始まる区間へ全量を入れる。窓はnominalな音声時間の集計であり、
workerを実際に100 msずつbatch実行へ変更してはいない。
未受信の空区間の費用を推定せず、部分窓・欠落を含む非空窓と、期待する全frameを受理した窓を
別集計する。最後の部分窓もEOFで確定する。epoch変更時は開いていた旧窓を閉じる。

処理件数・累積時間・最大時間・閾値超過数に加え、全受理frameと全非空窓の有界histogramを保持する。
公開reportが途中のsnapshotを間引いても、計数器は処理を省かない。
各binの上端はmicrosecondで100／250／500／1,000／2,000／4,000／8,000／16,000／25,000／
50,000／75,000／100,000／200,000／500,000／1,000,000、最後は上限なしである。
比較と加算はnanosecondで行う。p99は全計測母集団のceil(0.99*N)順位を含むbinの区間で報告し、
bin上端を正確なp99へ読み替えない。公開までの時間にはproducerのpublishからのqueue待ちも含む。

合成時計で、9／10-hop境界、1 frame欠落、epoch変更、部分EOF、拒否入力、閾値ちょうどと
その1 ns超過、最終binへのoverflowを検査した。窓費用の合計が全受理frame費用と等しいこと、
histogram件数と超過数も照合した。既存の実worker拒否入力検査では受理2・拒否9・EOF1の計数を確認した。

測定条件は `target/i10-worker-resources-20260918/plan.json` に取得前に固定した。
従来のmixed／sustain／stairに加え、4／16／64個のsine sustain Voiceを同時に生成した。
周波数110／220／330／440 Hzを反復し、各ampは0.04/sqrt(N/4)、seedは20260918、
sustain 6秒、release後0.5秒とした。各条件で期待する生存Voice数と固有onset Voice数を確認した。
これは現在接続されているpassive経路の負荷であり、未接続の共同推論や新しいcontextual decisionを
実行した測定ではない。身体profile転用も未受入のままである。

AMD Ryzen 9 9950X（16 core／32 logical CPU）の同じ環境でrelease取得した。
6条件・両busの全6,960受理frame、752非空窓を計数し、このうち740窓は全frame受理だった。
拒否・配信drop・source missingはすべて0。EOF処理は12件を別計上した。

| 条件 | 最大100 ms区間wall time、habitat / presentation (ms) | 50 ms超過窓、habitat / presentation | reportで確認した同時prototype対応の最大 |
| --- | ---: | ---: | ---: |
| mixed | 40.162 / 37.712 | 0 / 0 | 1 |
| sustain | 62.963 / 63.207 | 1 / 1 | 3 |
| stair | 63.225 / 63.147 | 1 / 1 | 3 |
| 4 Voice | 44.935 / 43.193 | 0 / 0 | 1 |
| 16 Voice | 44.646 / 44.077 | 0 / 0 | 2 |
| 64 Voice | 45.150 / 43.861 | 0 / 0 | 2 |

sustain／stairの最大窓はすべてsample [4,800,9,600) で、9 hopの総計約63 msのうち、
表refreshが約61 msを占めた。この区間での表構築が、現行workerの50 ms超過の主因である。
両素材の窓p99は(50,75] ms、その他の条件は(25,50] msのbinに入った。
4／16／64 Voiceの結果を全8 prototype対応や全I10資源受入へ拡張しない。

公開までの最大wall timeは36.990–145.420 msで、25 ms超過も残った。ただし今回は時間をpacedに
供給しないdeterministic offline rendererであり、この値をlive deviceの遅延分布へ流用しない。
全audio hop合計8.5333 ms、decision合計1.6 ms、device underrun／callback errorの受入も未実施である。

前後の音声6ファイルはbyte一致し、共通647表・656観測も数値と支持が一致した。
通常3素材と4 Voiceは、実指令・身体default・既存予測を含む従来の厳密な記録比較も通過した。
16／64 Voiceでは厳密比較が失敗した。差は到着時刻だけで、16 Voiceで40 field、64 Voiceで226 field。
身体descriptorの `available`、descriptor予測の `available_at_sample` と観測側の `available`、
発行時 `input_support` の第3値（実際の利用可能時刻）に限定された。
原窓・値・mask・予測値・教師値・誤差・学習状態は一致し、各時刻の因果的順序も維持した。
この差を消して「全記録一致」とせず、`strict_pass=false` と全差分を成果物へ保存した。
一回の比較から、計測追加の影響と自然なthread schedulingの変動は分離していない。

初期の検査器は、まだframeを受信していないWaiting snapshotの計数不在を誤判定した。
受理・拒否とも0、frame IDなしのWaitingだけを除外し、処理後の計数欠落は拒否する形に修正した。
到着時刻の差による厳密比較失敗は別記録として残し、数値・因果性検査と分離した。
histogram件数欠落、窓への費用算入漏れ、EOF未確定、閾値超過数矛盾の4変異も拒否した。

再現例:

```bash
cargo build --release --bin conchordal-render
python3 target/i10-worker-resources-20260918/acquire.py
python3 target/i10-worker-resources-20260918/verify.py
```

全体テスト1,160成功、0失敗、37 ignored、31 suite。
`cargo test exit=0 @ 2026-09-18T12:56:48+09:00`。Clippy、全target check、fmt、release build、
Python構文検査も通過した。source・入力・binary・取得command・計数検査・全体test log／statusを
hash付きで登録した。

I10全体は未完。通常共同推論・全head・実身体条件付けと候補／default消費が未接続であり、
全負荷受入も終わっていない。今回の再点検では、観測queueの実装が64件、§9.1の参照仕様が32件である
不一致も残項目として確認した。次の接続・資源検査で、この上限と到着時刻の挙動を扱う。
I9保留、I11作用無効は維持する。

### 観測queueの上限を参照仕様の32件へ修正（2026-09-18）

前節で確認した `observation::QUEUE_CAPACITY` の64件を、§9.1の32件へ修正した。
変更は各busの有界channel容量であり、live送信は従来どおり `try_send` で満杯時に新しいframeを
拒否する。deterministic offline rendererの送信は、従来のbackpressureを維持する。
EOFはanalysis側の終了経路から届け、audio callbackから待機・joinする経路は追加していない。

飽和検査を両busと「そのままEOF／受信を再開してからEOF」の4条件へ拡張した。
実channelの容量を参照値32と直接比較し、snapshot公開を停止した状態で256 frameをlive送信する。
送信が戻り、queue長は32以下、拒否数は少なくとも223件となることを確認する。
queue外で処理中の1件は、この32 slotとは別に存在し得る。
最終全体検査では各条件とも最初の32 frameを受理し、224 frameを拒否した。
復帰条件ではqueueの排出完了を確認してからframe 256を追加し、最終受理数は33件になった。

EOF後の入力時計、受理数＋拒否数、配信drop数を照合した。EOFまで復帰しない条件では、
最後の受理sampleから入力末尾までの差がdrop数×hopと一致した。復帰条件では最後のframe IDが256、
支持終端が257×128 sampleとなる。同じepochを維持し、source acquisition missingと
配信dropを混同しない。入力拒否の検査にも誤って加算しない。

前節の6条件とrelease binaryを固定し、容量32のreleaseで再取得した。
成果物は `target/i10-queue32-20260918/` に保存する。これは32件を保持した通常取得の比較であり、
上記の強制飽和検査とは別の証拠である。

6,960受理frame・752非空窓・740全frame受理窓という母集団は維持され、通常取得の配信drop・
source missing・入力拒否は0だった。音声6ファイルはbyte一致し、共通646表・655観測も一致した。
通常3素材と4 Voiceは既存記録の厳密比較を通過した。16 Voiceは3 field、64 Voiceは225 fieldの
到着時刻差が残り、厳密比較は不合格として保持した。差は前節と同じavailability欄だけであり、
原窓・値・mask・予測値・教師値・誤差・学習状態と因果的順序は一致した。

| 条件 | 容量32での最大100 ms区間wall time、habitat / presentation (ms) | 公開までの最大wall time、habitat / presentation (ms) |
| --- | ---: | ---: |
| mixed | 40.727 / 37.820 | 67.408 / 37.121 |
| sustain | 64.591 / 62.638 | 92.844 / 91.109 |
| stair | 65.456 / 62.322 | 79.009 / 71.720 |
| 4 Voice | 43.501 / 45.050 | 41.753 / 42.891 |
| 16 Voice | 43.354 / 43.569 | 41.598 / 41.839 |
| 64 Voice | 44.434 / 43.602 | 42.660 / 41.661 |

全条件の公開までの最大時間は145.420→92.844 msとなった。一方、sustain／stairの両busで
50 ms超過が各1窓残り、最大worker窓は65.456 msだった。容量修正を計算費用の解消やlive遅延の
合格とは扱わない。paired取得は引き続き時間をpacedに供給しないoffline rendererの一回比較である。

4／16／64個の生存Voice数と固有onset数を確認した。ただし16 Voiceでは、今回の間引きreportに
prototype対応がある瞬間が残らず、公開表から数えた最大対応数は0となった。
累積の表処理計数には実行費用が残っており、対応が一度もなかったという意味ではない。
対応数の最大値は、現在の公開snapshotだけでは全実行の最大値を証明できない。
他の条件も含め、全8 prototype負荷と全I10資源受入は未証明のままとする。

再現例:

```bash
cargo build --release --bin conchordal-render
python3 target/i10-queue32-20260918/acquire.py
python3 target/i10-queue32-20260918/verify.py
```

全体テスト1,160成功、0失敗、37 ignored、31 suite。
`cargo test exit=0 @ 2026-09-18T13:10:52+09:00`。Clippy、全target check、fmt、release build、
Python構文検査も通過した。取得前に固定した入力、変更source、前後binary、取得command、
検査結果、全体test log／statusをhash付きで登録した。

queue上限の不一致は解消した。I10全体は未完であり、共有表を主因とする50 ms予算超過、
到着時刻の変動、全実行のprototype負荷把握、通常共同推論・全head・実身体条件付けと
候補／default消費、評価時刻、liveおよび全audio-hop受入を残す。I9保留とI11作用無効は維持する。

### 全構築のprototype負荷集計とarticulation設定の参照渡し（2026-09-18）

間引きreportが瞬間的な対応を取り逃す問題に対し、共有表の実構築ごとに対応prototype数0〜8の
9 bucketを累積する `builds_by_assigned_prototypes` を追加した。再利用・延期・拒否は構築数へ
加算しない。prototypeとgroupは一対一とは限らず、ここで数えるのは対応があるprototypeの数である。
固定9個のu64でresident tableは72 byte増え、heap確保は増えない。reportとUIへ公開する。
通常時計の今回の取得では、bucketの総和がrebuilds、個数による加重和×7 class×32時刻が
attempted_cellsと一致した。u64末尾で未来時刻がoverflowする境界には後者の等式を適用しない。
既存Rust検査では再構築・延期・拒否後の累計を検査し、実取得では単調性、公開表との対応、
構築がない区間での不変性も照合した。総数を壊す変異と、総数を保って負荷を移す変異を拒否した。

観測と候補の `gesture::rates` は `TemporalGestureConfig` を値渡ししていた。
変更前のrelease binaryの両呼出し点で、1,496 byte (`0x5d8`) のmemcpyを確認した。
引数を参照へ変え、両経路の呼出し直前のコピーを除いた。rateの式、係数、演算順は変えない。
変更前後の逆アセンブルを保存した。この修正は他の設定コピーをすべて除去したという主張ではない。

成果物は `target/i10-load-census-20260918/` に保存する。前節の6条件を再取得し、さらにsustainの
articulation係数だけを非ゼロの固定値へ変更した対照を、保存済み変更前binaryと変更後binaryで
取得した。非ゼロ係数はelapsed・missing成分も含む工学的な回帰対照であり、適合済み認知モデルではない。
ゼロ係数条件との比較では観測348セル、投影464セルの状態確率が1e-6超変化し、最大差は0.029187だった。
係数が実際に両経路へ作用することを確認した上で、同一係数の変更前後を照合した。

変更後7条件・両busの8,182受理frame、884非空窓、870全frame受理窓、828構築を検査した。
配信drop・source missing・入力拒否は0。音声7比較はbyte一致、共通780表は72 byteの容量差を
除いて一致、共通789観測も一致した。通常3素材、4 Voice、非ゼロ係数対照の既存記録は厳密比較を通過した。
16 Voiceは40 field、64 Voiceは227 fieldのavailability差があり、厳密比較は不合格として保持する。
差の許可範囲は前節と同じ到着時刻欄に限り、値・mask・予測・教師・誤差・学習状態と因果的順序を照合した。

| 条件 | 全構築からの最大対応prototype数、habitat / presentation | 最大100 ms区間wall time、habitat / presentation (ms) |
| --- | ---: | ---: |
| mixed | 2 / 2 | 34.986 / 32.890 |
| sustain | 3 / 3 | 56.073 / 55.894 |
| stair | 3 / 3 | 57.542 / 55.999 |
| 4 Voice | 2 / 2 | 40.183 / 39.399 |
| 16 Voice | 2 / 2 | 39.985 / 38.759 |
| 64 Voice | 2 / 2 | 39.451 / 40.922 |
| sustain・非ゼロ係数 | 3 / 3 | 67.135 / 68.096 |

16 Voiceは両busとも62構築のうち61回が0対応、1回が2対応だった。公開表だけでは今回も最大0に
見えるが、全構築計数では2と確定した。4 Voiceも公開最大1に対し実構築最大2だった。
今回の母集団に4〜8 prototype対応はなく、全8 prototype負荷の受入は証明していない。

既存6条件の最大worker窓は65.456→57.542 ms、非ゼロ対照は75.461→68.096 msとなった。
7条件では50 ms超過が6窓、公開までの最大wall timeは100.076 msだった。
これは同一入力のoffline取得であり、反復統計による性能保証やpaced live遅延の合格ではない。
設定コピー削減後も全worker 50 ms予算は未達である。

全体テスト1,160成功・0失敗・37 ignored、31 suite。
`cargo test exit=0 @ 2026-09-18T13:20:01+09:00`。Clippy、全target check、fmt、release build、
Python構文検査が通過した。費用計数4変異とprototype計数2変異を拒否した。
固定入力、実行command、変更source、前後binary、逆アセンブル、比較元、検査logをhash付きで登録する。

全構築の負荷把握は接続できた。次は50 ms超過を生む共有表計算を絞り、最大対応数を含む負荷を検証する。
I10全体は未完であり、通常共同推論・全head・実身体条件付け・候補／default消費・評価時刻の確定、
全Voice負荷、liveおよび全audio-hop受入を引き続き残す。I9保留、I11作用無効を維持する。

### 候補特徴窓に重なるprofile frameだけを生成（2026-09-18）

共有表の非空候補窓では、2秒と250 msの各特徴集計へ毎回4秒分の375 frameを渡していた。
`window::summarize` は窓外の値を使わないが、iteratorから取り出す際のraw特徴・背景shareの
生成は窓外でも実行されていた。今回、`Phrase::project_action_window` のこの二つの入力だけを
窓に重なるprofile frameへ制限した。観測履歴はそのまま渡し、一般のwindow集計器は変更していない。

profile loaderが48 kHz・512 sample刻み・375 frameを検証し、この候補経路もhop 512を要求する。
この固定時計に対し、最初のframeを `floor(max(0, start-issue)/512)`、排他的な末尾を
`ceil((end-issue)/512)` とする。左端と右端に部分的に重なるframeを保持し、時刻がhop境界と
一致する場合だけ外側のframeを除く。iteratorのtake／skipで選び、集計する値の順序は変えない。
resident state、heap確保、profile binding、articulation、residual、accent、arrivalの入力経路は増減させない。
4秒端の例では、2秒窓は188 frame、250 ms窓は24 frameで足りる。発行時の空窓は従来どおり空のままである。

既存Rust検査を拡張し、発行時、1／511／512／513 sample、250 ms境界の前後、2秒境界の前後、
4秒端の前後を含む13終端×2窓を、全375 frameを渡す参照計算と照合した。
長窓は値・出所・観測／投影支持率のserialized結果、短窓はphraseが消費する4座標を比較した。
変更前後binaryによる通常取得は `target/i10-window-slice-20260918/` に保存する。
前節と同じ実音6条件と非ゼロ係数対照を使い、前節の変更後reportを比較元にした。

7条件・両busの8,182受理frame、884非空窓、870全frame受理窓、828構築を検査した。
全構築のprototype対応histogramは比較元と一致した。音声7比較はbyte一致、共通784表は容量を含め
完全一致、共通799観測も一致した。6条件は既存記録の厳密比較を通過した。
64 Voiceのみ219 fieldのavailability差が残り、厳密比較は不合格として保持する。
値・mask・予測・教師・誤差・学習状態と因果的順序は一致した。配信drop・source missing・入力拒否は0だった。

| 条件 | 最大100 ms区間wall time、habitat / presentation (ms) |
| --- | ---: |
| mixed | 34.432 / 33.667 |
| sustain | 56.459 / 54.829 |
| stair | 55.819 / 54.965 |
| 4 Voice | 38.956 / 37.906 |
| 16 Voice | 38.798 / 37.873 |
| 64 Voice | 39.795 / 38.509 |
| sustain・非ゼロ係数 | 65.328 / 64.719 |

既存6条件の最大worker窓は57.542→56.459 ms、非ゼロ係数条件は68.096→65.328 msだった。
7条件で50 ms超過が6窓残り、公開までの最大wall timeは95.202 ms。
削減幅は小さく、各条件が一様に速くなったという結果でもない。一回のoffline取得を性能保証や
live受入には読み替えない。全8 prototype対応も引き続き未検証である。

全体テスト1,160成功・0失敗・37 ignored、31 suite。
`cargo test exit=0 @ 2026-09-18T13:32:31+09:00`。Clippy、全target check、fmt、release build、
Python構文検査、6件の計数変異拒否も通過した。固定入力・変更source・前後binary・比較元report・
取得command・検査logをhash付きで登録する。

I10全体は未完。次はarticulation経路の残る繰返し計算を調べる。
通常共同推論・全head・実身体条件付け・候補／default消費・評価時刻の確定と、全Voice負荷・
live・全audio-hop受入も残す。I9保留、I11作用無効を維持する。

### articulationのframe共通特徴を標準化して参照する（2026-09-18）

観測／候補のarticulationは、同じframeのrise・decline・flux・motionを各経路で再び標準化していた。
4値と欠測indicatorをframeごとに一度計算し、経路ごとのelapsedだけを内積の `x[5]` へ差し込む。
11項の積と加算順、softplus、遷移・枝刈り・未知質量の式は維持する。新しいresident cacheやheap確保はない。
欠測16通り×4状態×6 elapsed値の384条件で、元の全design計算とrateのbit一致を確認した。
非ゼロ係数・平均、0／微小deviation、signed zeroを含み、既存の非有限elapsed拒否も維持した。

最初の共通配列を値渡しする版は不採用とした。7条件の音声・数値は一致したが、sustainと
非ゼロ係数sustainを前後binaryで交互3往復すると、全12 bus比較で表の総費用が5.53〜7.33%増えた。
この版のsource・binary・検査・失敗判断は `target/i10-rate-input-reuse-20260918/` に残す。
現在の版は11個のf64配列を参照渡しし、elapsed座標を式中で差し替える。
成果物は `target/i10-rate-input-borrow-20260918/` に保存する。

同じ変更前binaryとの交互3往復では、全12 bus比較で総frame費用と総表費用が減少した。
sustainの総表費用は2.33〜3.98%、非ゼロ係数条件は1.57〜2.13%減少した。
この6組の音声・既存記録は厳密一致、共通707表も一致した。最大窓の改善は全比較では成立せず、
これを50 ms保証や一般的な性能保証とは扱わない。

通常7条件の取得では、音声7比較、共通783表、共通793観測が一致した。
8,182 frame・884非空窓・870全frame受理窓・828構築と全prototype負荷histogramは維持した。
配信drop・source missing・入力拒否は0。16 Voiceは18 field、64 Voiceは233 fieldのavailability差が
残り、厳密比較は不合格として保持する。数値・mask・予測・教師・学習状態と因果的順序は一致した。
最大worker窓は既存6条件54.112 ms、非ゼロ係数65.637 ms。50 ms超過は6窓、最大公開時間98.133 msだった。

全体テスト1,161成功・0失敗・37 ignored、31 suite。
`cargo test exit=0 @ 2026-09-18T13:45:36+09:00`。Clippy、全target check、fmt、release build、
Python構文検査、6件の計数変異拒否も通過した。反復取得の初回は保存binaryの実行permission不足で
起動前に失敗したため、modeを復元して再実行した。失敗logを保存し、実行済み取得は重複させていない。

次の候補として、固定profileの完全一致する先頭raw frameを静的集計した。全軌道IDが異なっても、
前の完全hopまでの96 byte record列が一致する場合がある。各prototypeを単独で対応させ、全投影が
成功すると仮定した集計では、7 prototypeで必要hopが12,047→1,189、残る無音prototypeは405のままだった。
これは実workerの削減量ではなく再利用の候補量である。実装にはgroup／issue隔離、部分hopの再計算、
先頭の不一致と退役時の失効を保持する必要がある。次はこの完全一致prefixを対象にする。

I10全体は未完。50 ms・全8 prototype・全Voice負荷、通常共同推論・全head・実身体条件付け・
候補／default消費・評価時刻の確定、live・全audio-hop受入は引き続き残る。I9保留、I11作用無効を維持する。

### 異なる軌道の完全一致raw prefixをarticulationで再利用（2026-09-18）

全軌道IDが異なる候補間でも、前の評価までの完全hopが同じ場合にarticulationの途中状態を再利用する。
共有表の構築中、保存済みprefixと新しい軌道の該当frameを比較し、maskと全11値のf64 bitが
すべて一致した場合だけtrajectory IDを付け替える。signed zeroも区別し、近似距離やhashだけでは判定しない。
その後、既存のgroup・issue・評価終端の検査を通して復元する。時刻巻戻しとgroup交代では復元しない。

cacheは一回のimmutableなTable構築に限る。同じgroupとissueでは発行直前の観測、背景energy、
Gestureの設定も固定されるため、同じraw prefixは同じ変換済み入力を与える。新しいissueやmodelへ
保存状態を持ち越さない。完全hopの状態だけを保存し、前回の終端で切った部分hopは次の軌道で再計算する。
特徴窓、残差、accent、arrival、raw head、Cell自体の同一軌道再利用は変更していない。
既存448 byteのProjectionCacheを使い、resident fieldとheap確保は追加していない。

合成profileでは31個の異なるfuture tailと、途中に先頭不一致を持つ軌道を追加した。
通常Tableの224セル位置を直接計算と照合し、支持された全セルのarticulationを確認した。
実計算870 hop・再利用11,160 hopを記録した。さらに非ゼロelapsed係数を使う14条件で、
部分hop直後の別軌道、先頭不一致、巻戻し、group世代交代を直接計算とserialized比較した。

成果物は `target/i10-cross-trajectory-prefix-20260918/` に保存する。
既存7条件・両busの8,182 frame、884非空窓、870全frame受理窓、828構築を検査した。
全構築のprototype対応数と候補セル数・直接呼出し数は前版と一致した。
articulationの実計算は4,915,279→496,073 hopとなり、削減した4,419,206 hopは再利用計数の増分と一致した。
音声7比較はbyte一致、共通781表は全field一致、共通790観測も一致した。
16 Voiceの18 field、64 Voiceの265 fieldにavailability差が残り、厳密比較は不合格として保持する。
それ以外の既存記録は厳密比較を通過し、数値・mask・予測・教師・学習状態と因果的順序は全条件で一致した。
配信drop・source missing・入力拒否は0だった。

| 条件 | 最大100 ms区間wall time、habitat / presentation (ms) |
| --- | ---: |
| mixed | 9.236 / 7.256 |
| sustain | 17.560 / 17.092 |
| stair | 17.547 / 17.367 |
| 4 Voice | 12.727 / 12.412 |
| 16 Voice | 12.457 / 12.899 |
| 64 Voice | 13.239 / 13.272 |
| sustain・非ゼロ係数 | 18.647 / 18.088 |

この取得母集団では50 ms超過は6→0窓、最大worker窓は65.637→18.647 msとなった。
公開までの最大wall timeは16.894 msで、25 ms超過も0だった。
sustainと非ゼロ係数sustainを前後binaryで交互3往復した比較でも、全12 bus比較で総表費用が減少した。
削減は前者73.44〜73.83%、後者75.70〜76.68%。6組の音声・既存記録は厳密一致、共通696表も一致した。
反復取得での変更後最大worker窓は19.555 msだった。これはofflineの現在経路に対する測定であり、
全8 prototype対応、全I10処理、paced liveや全audio-hopの受入を意味しない。

全体テスト1,162成功・0失敗・37 ignored、31 suite。
`cargo test exit=0 @ 2026-09-18T13:55:55+09:00`。Clippy、全target check、fmt、release build、
Python構文検査、6件の計数変異拒否も通過した。変更source・前後binary・固定入力・比較元・19取得command・
検査logをhash付きで登録する。

今回の7条件で見えていた共有workerの50 ms超過は解消した。次は最大対応数を含む負荷を検証し、
通常共同推論と候補消費の未接続部分へ進む。全head・実身体条件付け・候補／default消費・評価時刻の確定、
全Voice負荷・live・全audio-hop受入を残し、I10全体は未完とする。I9保留、I11作用無効を維持する。

### 全8 prototype対応の共有表負荷を明示取得（2026-09-18）

前節の実音素材は最大3対応だったため、凍結済み8 profileを使う明示負荷試験を追加した。
`action_profiles::tests::saturated_action_profile_table_load` は通常テストではignoreし、環境変数で
固定configと出力先を指定してrelease test binaryから実行する。成果物は
`target/i10-eight-prototype-load-20260918/` に保存する。instrumentへの設定や出力経路は追加していない。

入力は因果的な時計を持つ合成raw frontend recordであり、PCMからの取得ではない。
全8 profileを1音群または7音群へ巡回的に割り当てる。指定groupとの標準化距離は実計算するが、
試験内だけ `f64::MAX` の距離門を使って占有を強制する。通常の0.25門、元のmedoid、raw profileは変更しない。
この対応を、実音の照合成功や身体候補の適格性へ採用しない。

両bus×1／7音群×通常音量／静音×ゼロ／非ゼロarticulation係数の16条件を取得した。
各条件は48 kHz・hop 512で201 frame、10 hopごとの21構築を含む。全336構築で対応数8、
各構築1,792試行位置・1,032有効セルを確認し、総試行位置は602,112だった。
各条件の最終表を再利用なしの直接計算と比較し、合計28,672位置が一致した。
この位置数には、元profileでbindingがない760位置／条件のNone照合も含む。

計時は `Gesture::advance`、`Phrase::advance`、`Table::refresh` の合計であり、合成入力生成、
強制割当、serialization、初期化、最後の直接照合を除く。音響frontend、memory、section、queue公開、
audio処理も含まない。100 msのsample窓へframe開始時刻で全処理費用を割り当て、各条件21完全窓と
1部分窓を記録した。以下は完全窓の最大wall timeである。

| 音群数 | 入力 | 係数 | habitat (ms) | presentation (ms) |
| ---: | --- | --- | ---: | ---: |
| 1 | 通常音量 | ゼロ | 40.416 | 40.334 |
| 1 | 通常音量 | 非ゼロ | 44.254 | 43.918 |
| 1 | 静音 | ゼロ | 24.811 | 24.823 |
| 1 | 静音 | 非ゼロ | 27.440 | 27.278 |
| 7 | 通常音量 | ゼロ | 49.970 | 49.810 |
| 7 | 通常音量 | 非ゼロ | 52.762 | 52.882 |
| 7 | 静音 | ゼロ | 31.648 | 31.641 |
| 7 | 静音 | 非ゼロ | 33.930 | 33.913 |

完全窓336件のうち17件が50 msを超過した。7音群・通常音量・非ゼロ係数のhabitatで9件、
presentationで8件だった。最大窓は `[4800,9600)` の9 frameで、gesture＋phraseが10.031 ms、
表が42.851 msを占めた。測定した3処理だけで予算を超え、除外したworker段階の余裕も必要である。
全8対応の表計算を取得したが、全worker資源受入は引き続き未達とする。

全条件・全構築でarticulationとcontinuation raw headの1,032セルは支持された。
closure raw headの支持は0であり、matched arrivalや全headを含む最大負荷の検証にも達していない。
原窓、完全窓のframe数、全8対応histogram、1,792試行位置、1,032有効セル、group数、質量保存を
独立Python検査で照合し、prototype欠落・case欠落・時計違反・セル数違反の4変異を拒否した。

通常全体テスト1,162成功・0失敗・38 ignored、31 suite。
`cargo test exit=0 @ 2026-09-18T14:08:39+09:00`。今回のignore 1件はreleaseで明示実行し、成功した。
Clippy、全target check、fmt、release test build、Python構文検査も通過した。
config・profile・source・test binary・実行command・計測結果・全体test logをhash付きで登録する。

次は7音群でのgesture／phraseと共有表の費用を絞る。通常の照合成立、closure等の未支持head、
全worker・全Voice・live・全audio-hop、通常共同推論・実身体条件付け・候補／default消費・評価時刻は
残る。I10全体は未完、I9保留、I11作用無効を維持する。

### phraseの生存確率だけを使う経路から退出計算を除去（2026-09-18）

全8対応試験で残った費用に対し、`phrase::law` の呼出し用途を分けた。観測と候補のcontinuationは
状態混合後の生存確率だけを使うため、4退出のlogit・指数・正規化を計算しない。phraseの遷移と
通常forecastは退出計算を維持する。両用途で同じ26座標、hazard内積の加算順、証明付き積分、
状態順の生存確率加算を使う。積分不能は欠測のまま、退出logitの失敗は生存確率の支持を失わせない。
完全計算側では入力vectorの二重構築も除き、積分後はelapsed座標だけを更新する。
新しい常駐状態・heap確保は追加しない。

先行するvector再利用だけの版は数値一致したが、最大窓52.556 msで超過を解消しなかった。
成果物を `target/i10-phrase-vector-reuse-20260918/` に保持する。時間差は小さく符号も変わり、
3組目の反復は後続版のcompileと重なった可能性があるため、単独の速度改善の根拠には使わない。
最終版の素材・前後binary・実行記録・検査は `target/i10-phrase-survival-evaluation-20260918/` に保存した。

欠測4パターン、5単独状態と混合・全ゼロ質量、8時間区間、4hazard傾きの896条件で、完全計算と
生存確率専用計算のbit一致を検査した。退出logitがoverflowする追加条件でも支持と値が一致した。
最終版の全8対応16条件は、変更前の336表と全数値・mask・支持・非時間資源カウンタが一致した。
全構築で1,792試行位置・1,032有効セルを維持し、最終表の直接計算との28,672位置比較も通過した。

初回取得の完全窓336件では、最大52.882→48.908 ms、50 ms超過17→0件となった。
同じ固定入力を変更前後のrelease test binaryで交互3往復し、48条件対すべてで測定対象の総費用が
3.58〜9.09%減少した。反復の変更後1,008完全窓も超過0件、最大48.994 msだった。
3組の共通1,008表も全数値が一致した。計時範囲は前節と同じgesture・phrase・表構築だけであり、
50 msに対する約1 msの残余で、未測定のworker処理を収容できるとは判定しない。

通常rendererの7素材も音声がbyte一致、共通819表と831観測が一致した。
mixed、sustain、stair、4／16 Voice、非ゼロ係数sustainの既存記録は厳密一致した。
64 Voiceではavailabilityに229座標の差があり、厳密比較は不合格のまま保持する。
それ以外の数値・mask・予測・教師・学習状態は一致し、availabilityの因果順序も通過した。
8,182 frame・884非空窓の最大worker時間は18.125 ms、公開までの最大時間は16.419 msだった。
50 ms窓超過、25 ms公開超過、drop、入力欠落、rejectはいずれも0だった。

全体テスト1,163成功・0失敗・38 ignored、31 suite。
`cargo test exit=0 @ 2026-09-18T14:28:01+09:00`。Clippy、全target check、fmt、release build、
release test build、Python構文検査を通過した。既存の資源・負荷検査に対する10変異も拒否した。
最終版の7 renderer取得と7明示assay実行、source・固定入力・比較元・binary・検査結果をhash登録した。

全8対応で測った3処理の予算超過は今回の条件で解消した。強制対応を通常の照合成功とは扱わず、
closure支持0、matched arrivalと全headの負荷未検証も維持する。次はこれらの負荷と全workerの
余裕を確認し、通常共同推論・path寿命・条件付きscore・実身体候補／default消費へ進む。
評価時刻、全Voice・live・全audio-hop受入は残る。I10全体は未完、I9保留、I11作用無効を維持する。

### closure支持とarrival入力を含む全8対応負荷（2026-09-18）

前節の試験はmemory予測とarrival状態を渡していなかったため、未来残差とclosureの計算負荷を
含んでいなかった。既存の明示負荷試験へ `CONCHORDAL_I10_MAX_LOAD_HEADS=1` を追加し、
支持なしの対照16条件と、支持あり16条件を同じrelease test binaryで取得した。
成果物は `target/i10-supported-head-load-20260918/`、登録はfeature manifestの
`i10_supported_head_synthetic_load`。変更は試験と `cfg(test)` のfixture公開だけであり、
通常の計算、profile、0.25照合門、評価時刻、生成作用は変更していない。

支持あり条件では、fluxを16 hop周期で0.125から4へ上げ、固定profileの標準化と通常のsalience・
peak・weight式から4-hop支持付きdetector recordを作る。同じ検出結果をphraseとrecurrenceへ渡す。
recurrenceの音群所有だけを試験内で固定し、ledger・period・grouping・arrivalは実装を通す。
期待区間は第2hop以降、直前に受理したdescriptorを一定継続すると仮定して作り、今回hopの値を
先読みしない。発行時計は直前hop終端、期待終端は今回hop終端+4秒とする。最初のhopの残差は欠測。
これは合成raw入力と合成検索結果であり、実PCMでの音群成立・検出・検索成功を示さない。

試験作成時に、accentをrecurrenceだけへ渡す不整合を修正した。また最初の明示実行は、
`observed_prefix` にreceipt時刻を入れた試験データをledgerが拒否し、exit 101となった。
通常detectorと同じ中央hop終端へ修正し、source開始も4つのraw frameの重なる解析支持を含む
`max(0, receipt-3584)` とした。失敗source・binary・入力・command・logを `failed-prefix/`、
中間の数値照合を `debug-reference/` に保持し、最終版の結果とは分けた。

最終版では両modeとも336構築・602,112試行位置・28,672位置の直接比較を通過した。
対照の336表と非時間資源カウンタは前節の保存結果と一致した。
支持ありでは初回構築を除く320構築で、closureとcontinuationのraw headが各1,032セルで支持された。
Pythonで支持時計、直前descriptor、観測残差、arrival原状態を照合した。さらに両busを含む8素材を
既存の独立profile・閉形式hazard・ordinal参照へ通し、重複除去後2,352セル、未来残差1,280セル、
closure分布2,240件、continuation分布34,104件を照合した。この簡略reportには通常phrase全snapshotが
ないため、通常marginalとのraw mixture整合比較は今回の独立参照に含めない。

計時範囲はgesture・phrase・表構築に、支持あり時のfixture初期化・所有設定とledger・period・
grouping・arrival更新を加えたもの。後者は `recurrence_ns` としてinference時間の内数でも記録した。
raw／memory入力の組立、強制対応、serialization、直接照合は除く。支持ありはpulseとdetector入力も
異なるため、対照との差をclosureだけの費用とは解釈しない。音響frontend、実検索、section、queue、
全audio処理を含むworker全体の測定でもない。

対照336完全窓の最大は48.172 ms、50 ms超過0件。支持あり336完全窓では最大52.420 ms、超過16件。
7音群・通常音量・非ゼロ係数の両busで8件ずつだった。最大窓 `[38400,43200)` はinference 10.737 ms、
表41.683 msである。前節の限定条件では超過を解消したが、今回の広い負荷では再び予算未達となった。

arrival原状態は支持されても、全構築で未来の点予測セルは0だった。固定profileの全bindingを別途
走査すると、有効1,032セルのうち40セルが発行時点、992セルが未来だった。未来992セルすべてで
評価窓内の候補accentは0、960セルはhop途中で窓が終わり、残り32セルもreceipt不明を解消しなかった。
原arrival履歴を与えるだけではこの欠測を解消できない。invitationや共同再重み付けを含む全headの
負荷達成とは扱わない。

評価時刻の判断材料として、採用前の静的比較も保存した。動作250 ms後を次の512-sample境界へ
切り上げる場合、1,032セル中131セルでarrival点予測が成立し、64セルは4秒profile外となった。
1秒後なら101セルで成立し、256セルがprofile外となった。範囲外は欠測のままとし、profileを延長したり
評価時刻を通常コードへ採用したりしていない。これは固定profile上の支持数の比較であり、ordinal校正、
候補の音楽的有用性、変更後のworker負荷を証明しない。評価時刻の選択を作者へ提示した。

最終sourceの全体テスト1,163成功・0失敗・38 ignored、31 suite。
`cargo test exit=0 @ 2026-09-18T14:58:36+09:00`。Clippy、全target check、fmt、release test build、
最終2取得の明示test、Python独立参照、7件の変異拒否を通過した。source・config・profile・test binary・
取得command・原窓・数値参照・失敗記録をhash付きで登録する。

次は支持あり負荷の費用削減と、判断後の評価時刻に沿った帰結窓の検証を分けて進める。
通常共同推論・path寿命・条件付きscore・実身体候補／default消費・全head・全worker／Voice・live・
全audio-hop受入は残る。I10全体は未完、I9保留、I11作用無効を維持する。

### 同じforeground開始時刻の候補continuation計算を再利用（2026-09-18）

前節の支持あり負荷に対し、同じセル内で重複するphrase継続計算を除いた。
保存済みの重複除去後2,352セルでは、15経路に対してforeground開始時刻は3〜8種類だった。
4／5経路の初期セルも含めると、34,104経路評価に対して異なる開始時刻は12,712件だった。
この標本の差21,392件は全表の実測呼出しカウンタではなく、保存snapshotから求めた再利用可能数である。

`Phrase::project_action_window` は既存の15出力配列を順に埋め、同じ `foreground_start` の
先行出力があれば生存確率・支持タグ・ordinal分布を再利用する。`None` も一つの別状態として比較する。
各経路のindex・issue massと配列順序、元のunknown massは維持し、経路自体を統合しない。
新しい常駐field、heap確保、セルをまたぐcacheは追加しない。
現在このセルのhead入力はforeground開始時刻以外が共通である。将来、別のpath条件付き入力を
導入する場合は、この再利用条件を拡張するか再利用を外す必要がある。

独立に各経路を再計算するmodule testを追加した。別ID・別credit・別質量を持つ15経路について、
2種類のhazard傾き（通常／積分不能）、articulationあり／なし、continue／release、発行時／完全未来hop／
部分未来hopの24条件・360出力を比較し、数値・欠測・観測／予測タグ・metadataが一致した。
処理後の元経路の質量・ID・credit・開始時刻も保持された。

成果物は `target/i10-continuation-duration-reuse-20260918/` に保存した。
支持なし／ありの計32条件で、変更前後の672表snapshotの全fieldと非時間資源カウンタが一致した。
各modeの最終表は、再利用なしの直接セル投影とそれぞれ28,672位置を比較した。
支持ありの独立profile・hazard・ordinal参照8素材も通過した。
closure／continuation支持、arrival原状態、未来arrival点予測0という前節の範囲は変更していない。

初回の支持なし336完全窓は最大39.297 ms、支持あり336完全窓は最大43.631 msとなった。
前節の支持あり最大52.420 ms・50 ms超過16件に対し、今回は超過0件だった。
支持あり入力を変更前後のbinaryで交互3往復した48条件対でも、全てで測定対象の総費用が
6.82〜22.31%減少した。変更後1,008完全窓は最大43.875 ms・超過0件、共通1,008表snapshotも一致した。
計時範囲は前節の合成recurrence部分・gesture・phrase・表構築であり、全workerの受入ではない。

通常rendererの7素材も音声がbyte一致、共通818表snapshotと826観測が一致した。
mixed、sustain、stair、4／16 Voice、非ゼロ係数sustainの既存記録は厳密一致した。
64 Voiceはavailabilityに183座標の差があり、厳密比較の不合格を維持する。それ以外の数値・mask・
予測・教師・学習状態は一致し、availabilityの因果順序も通過した。
8,182 frame・884非空窓の最大worker時間は13.136 ms、公開までの最大時間は11.428 msだった。
50 ms窓超過、25 ms公開超過、drop、入力欠落、rejectはいずれも0だった。

全体テスト1,164成功・0失敗・38 ignored、31 suite。
`cargo test exit=0 @ 2026-09-18T15:12:35+09:00`。Clippy、全target check、fmt、release build、
release test build、Python構文検査、13件の資源・支持変異拒否も通過した。
7 renderer取得と8明示assay実行、source・前後binary・固定入力・比較元・数値参照・test logをhash登録した。

支持ありの現負荷について余裕を確保した。次は通常共同推論の所有・path寿命・条件付き特徴とscoreの
接続へ進む。評価時刻の選択は未回答のまま保持し、時刻変更を前提にしない作業を進める。
実身体候補／default消費、全head、全worker／Voice・live・全audio-hop受入は残る。
I10全体は未完、I9保留、I11作用無効を維持する。

### 共有文脈の4候補を共同正規化へ渡す遷移核（2026-09-18）

通常経路との照合で、共同推論の正規化核と局所tuple合成は存在するが、共有候補の遷移核が
未実装と確認した。`joint/shared.rs` に継続・不明・検索済み・新規／対照の4固定slotを追加した。
入力は呼出側が選んだ最強の支持候補であり、今回の核自体は実音からの候補選択を行わない。
opaque keyは現在の文脈ラベルではなく完全な経路を指す契約とした。異なるkeyを統合しない。
同じkeyが複数roleに現れる場合は、正のscoreを持つ継続・検索済み・新規／対照の順で最初を残し、
そのscoreを使う。scoreを加算せず、空いたslotを弱い候補で埋め直さない。

既知候補を正規化し、合計を `exp(-dt/120)`、不明を `-expm1(-dt/120)` とする。
既知候補がなければ不明が1となる。`dt=0` でも質量0の不明slotは残す。
欠測時も既知親の採用済み遷移priorは維持するが、不明親は新しい観測支持なしに既知へ回復しない。
不明親に捨てた経路を継続として渡す入力は拒否する。score・時計の不正はmask適用前に拒否する。
配列のみを使い、heap確保や常駐状態は追加していない。

80桁Decimalによる確率空間の独立参照80条件と、module内4検査を通過した。
支持の組合せ、既知／不明親、観測／欠測、重複role、0・極小・長時間、極大／subnormal scoreを含む。
2／8／32／60秒の欠測を1／2／1,000分割した遷移質量も一致した。
7既知親と1不明親の31候補を、8音群の局所入力とともに既存の共同正規化核へ渡した。
非ゼロの合成観測scoreあり／なしの両方で全確率列挙と比較し、7保持・16剪定の既知経路と、
明示的不明の質量を分けて保持した。一般の保守的な共有候補上限32は変更していない。

全体テスト1,168成功・0失敗・38 ignored、31 suite。
`cargo test exit=0 @ 2026-09-18T15:33:21+09:00`。
Clippy、全target check、fmt、diff検査、Python構文検査も通過した。
登録は `i10-shared-proposals.json`、成果物は `target/i10-shared-proposals-20260918/` に保存する。

この核は既存のtest-only `joint` 配下にあり、通常観測や通常帰結表には未接続である。
実際の最強候補選択、完全な経路状態の所有・寿命、bus／epoch・観測支持・一回更新の管理は未実装。
完全な条件付き特徴表・scoreとheadへの接続も残る。今回の検査は実音上の保持・再認、
通常workerの実時間性、科学的妥当性を示さない。次はこの遷移入力を作る所有状態と候補選択を
通常観測へ接続する。I10全体は未完、評価時刻の選択待ち、I9保留、I11作用無効を維持する。

### 通常section経路の更新IDと親参照を公開（2026-09-18）

通常経路を照合すると、sectionの `context_id` は文脈名であり、同じ文脈に複数のphrase枝・
履歴が対応していた。これを完全な経路のIDとして共同推論へ渡すことはできない。
section Stream内の単調増加counterから、根と各更新後の経路へ一意な `path_id` を割り当てた。
継続でも新しい経路versionを作り、`parent_path_id` で直前を参照する。
bus・epoch・音群generationはGroupSnapshotのownerと組にする。文脈名と経路を統合せず、
剪定や公開順の並べ替えでもIDを振り直さない。counter上限は巻き戻さずエラーにする。

公開候補に現在の `phrase_path_index`、音群snapshotに `previous_end_sample` と最大16件の
親経路ID・親phraseスロットを追加した。これにより間引かれたreportでも、その更新が参照した
親inventoryとphrase子の親indexを照合できる。実際の履歴・pending所有・対応は従来のPathが保持する。
今回のIDはlocal sectionの更新versionであり、共有文脈の継続IDへ直接流用してはいけない。

さらに、その更新自身の `transition` と `transition_score` を公開した。
前者はstay／new_context／return／contrast／developmentであり、その後のstayにも残る
解釈ラベル `relation` とは区別する。後者は既存headの生存・退出確率と帰還先の支持質量から
計算した、親form質量を掛ける前のraw scoreである。共同観測potentialや校正済み確率ではない。
既存のweightの乗算順、候補順位・剪定・head数値は変更していない。
追加は固定長の親inventoryとscalar metadataで、新しいheap確保箇所はないが既存structは大きくなる。

既存Rust検査2件を拡張した。通常Phrase→Sectionの46更新では、欠測をまたぐ場合も含め、
全親inventoryが直前の保持経路と一致し、phrase親参照・ID非再利用・同じ文脈の別経路・
一親からの分岐を検査した。帰還fixtureではraw scoreと元のweight、履歴ラベルと今回の遷移の
違い、counter上限での拒否を検査した。section関連30検査が通過した。

変更前後のrelease rendererでsustain・stairを取得し、両busで異なる4,330経路と
4,234件のphrase親参照を検査した。0の工程確認用section係数について、stay／new／contrastの
raw scoreを閉形式で4,362件照合した（EOF等の反復snapshotを含む）。実音のreturn正例は
この取得ではなく、前述のmodule fixtureの検査範囲に留まる。
共通263観測の既存field・候補表、onset 2件、身体default 253件、身体descriptor 252件、
outcome／descriptor予測各4件、population step 1,259件とWAVは厳密一致した。
今回の取得にはavailability差もなかった。孤立した正の親ID、親phraseスロット違い、ID重複、
raw score改変の4つの保存記録対照を拒否した。

2,598 accepted frame・280非空窓の最大worker窓は12.899 ms、最大公開時間は10.626 msだった。
50 ms超過窓、drop、reject、入力欠落は0だった。これは2素材の現行workerの測定であり、
完全な共同推論・全Voice負荷・live audioの資源受入ではない。

全体テスト1,168成功・0失敗・38 ignored、31 suite。
`cargo test exit=0 @ 2026-09-18T15:42:41+09:00`。
Clippy、全target check、fmt、release build、diff検査、Python検証を通過した。
登録は `i10-section-path-identity.json`、前後binary・固定入力・4取得・検証・全test出力は
`target/i10-section-path-identity-20260918/` に保存した。

通常のlocal sectionに経路識別と遷移入力を接続したが、共有文脈を音群間で所有する状態は未実装。
最強共有候補の選択、共同特徴の全登録とscore更新、全head、実身体候補／default消費と
全資源受入が残る。次の接続では、既存の独立section質量を共同観測scoreへ読み替えず、
対応する親経路状態とraw遷移を使う。I10全体は未完、評価時刻の判断待ち、I9保留、I11作用無効を維持する。

### 一つのarticulation状態に条件づけたphrase headを共用（2026-09-18）

通常のphrase headは、articulationの各状態でheadを評価してから周辺化する。
共同候補が必要とするのは、その周辺化済みの値ではなく、候補経路が持つ一つの状態での値である。
`phrase::conditional_law` を分離し、通常の混合と共同候補の両方から同じ26座標のheadを
使えるようにした。`Option<gesture::State>` で一つの状態を指定し、`None` は四つの状態指示値を
欠測にする。呼出側がvaluesの先頭四座標に周辺確率を入れても、その値では状態を置き換えない。
残る座標の既存尺度・欠測maskは維持し、新しいstage-3特徴を追加していない。

通常の混合は既存の状態順・乗算順を維持する。生存確率だけを使う経路ではexit logitを計算せず、
hazardが評価不能な場合とexitだけが評価不能な場合も従来どおり区別する。
新しいheap確保はなく、既存の候補continuationの再利用条件も変更していない。

共同tuple側には既知の活動中foreground用 `List::active_phrase` を追加した。
呼出側が所有する五つの異なる経路ID（stay／new／overlap／reinterpret／inactive）を受け取り、
選択したarticulation状態の生存・退出scoreを生成する。inactiveには観測済み低energyのgateが必要。
採用した正の既知scoreを既存Listの契約で正規化し、30秒scaleの不明遷移を一度だけ加える。
head全体が評価不能なら不明1、exitだけが評価不能なら支持されたstayのみを既知候補とする。
これは通常bootstrapの `Group::step` と同じposteriorを作るという主張ではなく、
共同推論の採用候補正規化に従う別の遷移入力である。

経過時刻は整数sampleとsample rateで受け取り、不明遷移のdtには整数sample差を使う。
大きな浮動小数点経過時刻の引き算からhop幅を作らない。
逆向き時刻、sample rate 0、重複した候補IDは拒否する。

新規Rust検査2件を追加した。四状態と不明、特徴の有無、三つの区間、inactive gateの有無の
90条件を、定数softplus hazardの閉形式と確率空間のsoftmax・正規化で独立照合した。
状態別係数によって候補質量が異なること、無関係な周辺指示値を混入しても置換されないこと、
head評価不能、時刻・IDの不正、大きな経過時刻でも不明遷移が物理hop幅に従うことも検査した。
実際の条件付きphrase Listを他の成分Listとtuple合成し、stay・不明・四つの境界組合せと
16 heap pop／86評価の上限、遷移質量の正規化を検査した。
proposal関連7検査、既存の生存確率専用経路等を含むphrase関連25検査が通過した。

通常rendererはsustain、stair、状態別の非ゼロ係数を加えたstairの3条件を前後比較した。
追加係数は工程確認用で、適合や作者採用を行ったものではない。
両busの共通399観測では経路metadata・候補表を含む既存値が厳密一致し、既存イベント記録・
availability・WAVも一致した。係数なし／ありの対照間では141の共通生存予測すべてが変わり、
最大差は0.067027だった。この対照は非ゼロ係数の経路を通した証拠であり、予測精度の証明ではない。
変更後3,974 accepted frame・428非空窓の最大worker窓は12.802 ms、最大公開時間は10.538 msで、
50 ms超過窓・drop・reject・入力欠落は0だった。全負荷・live audioの資源受入は未実施である。

全体テスト1,170成功・0失敗・38 ignored、31 suite。
`cargo test exit=0 @ 2026-09-18T16:04:06+09:00`。
Clippy、全target check、fmt、release build、diff検査、最終Python検証を通過した。
比較scriptの初回構文エラーは修正し、失敗logも最終取得と分けて保存した。
登録は `i10-conditional-phrase-law.json`、固定入力・6取得・前後binary・検証・test出力は
`target/i10-conditional-phrase-law-20260918/` に保存した。

共同候補アダプタはtest-onlyであり、通常観測に共同posteriorを設置したわけではない。
inactive／unknown親の候補生成、共有・局所の完全な経路所有、共同特徴の全登録とscore更新は残る。
経路ごとに異なる値を通常continuationへ入れる際は、foreground開始時刻だけを使う既存cacheを
拡張するか外す必要がある。全head、実身体候補／default消費、評価時刻の判断、全資源受入も未完。
I10全体は未完、I9保留、I11作用無効を維持する。

### 帰結評価時刻の任意設定と比較、負荷超過と積分の支持欠落（2026-09-18）

帰結評価時刻の採用判断は保留を維持する。既定の共有帰結表は従来の正確な動作時点を使い、
比較用の明示設定 `temporal_action_profiles.evaluation_delay_ms` を追加した。
値は整数0–4000 msで、設定した場合だけ
`issued_at + 512 * ceil((action_offset + 48 * evaluation_delay_ms) / 512)` を使う。
48 kHz・512 sampleの発行時点相対gridであり、絶対sample 0を基準に丸めない。
明示0は次の完全hopへの切上げ、未指定は切上げなしである。
丸めた相対評価時刻が192,000 sampleを越える場合や整数overflowは評価不能とし、
終端への切詰め・未収録tailの外挿はしない。動作時刻そのものは変更しない。

設定をimmutable profileと表のmetadataに保持し、表の再利用・更新間隔判定にも含めた。
同じprofile hashでも設定が違えば表を作り直す。元の観測や凍結headを更新しない。
reportには設定したsample遅延と `evaluation_unavailable_cells`、UIには評価方針と評価不能数を表示する。
このcounterは全prototype/class/offset位置を数えるため、元からtrajectoryのない位置も含む。
新たなhop-path heap確保はない。保持容量は今回のbuildでprofileが16 byte、表が32 byte増えた。

新規Rust検査3件は、未指定設定の往復、設定値の拒否、絶対時刻に依存しない切上げ、境界とoverflow、
同一時点での設定切替・cache再利用・範囲外消去、観測不変を検査した。
初回の試験入力には100-sample fixtureと512-sample時刻の不整合があり、fixtureを修正した。
最大割当の既存ignored assayにも設定別の直接投影照合と最終arrival入力の出力を追加した。

通常rendererはsustain、stair、非ゼロphrase係数のstairを、変更前、未指定、250 ms、1秒の計12回取得した。
未指定の共通399観測では新counterと保持容量を除く既存候補表・観測が一致した。
各遅延設定と未指定の共通観測は各332件で、候補表以外の観測・既存イベント記録・availabilityが一致し、
全条件のWAVも一致した。対応する1,848セルのうち1,812セルでcontinuation分布が変わり、
最大カテゴリ差は0.0141864だった。時刻metadataだけを変えた比較ではない。
これらの短い素材にはfuture residual referenceがなく、その正例を検査したとはしない。
独立参照は原profileのaccent/arrival receipt、定数arrival hazardの支持条件、ordinal head、
係数なし二素材のphrase生存確率を照合した。観測residualの平均自体は独立再構成していない。
設定・評価終端・範囲外セル数を改変した三つの負例も拒否した。

最大割当は両bus、音群1/7、quiet/loud、weightedラベルfalse/trueの16報告行を三設定で測定した。
後の条件監査でgesture係数は両ラベルともゼロと判明した。非ゼロ係数対照の主張は末尾で訂正する。
各設定3,216更新、336表構築、直接照合は三設定計86,016位置、arrival receiptの独立照合は44,416セルだった。
全8 prototypeで有効セル数は未指定1,032、250 msで968、1秒で776となった。
元の有効bindingのうち、それぞれ0、64、256セルがprofile範囲外となる。
Gesture・Phrase・Recurrenceと表更新の合計最大は41.303、171.126、157.652 ms。
50 ms超過は未指定0、各遅延設定336回で、**遅延設定の最大割当は予算不合格**だった。
この測定は合成入力・強制割当で、matcher・section・全worker・Voice・実audio deviceを含まない。
通常rendererでも250 msのsustainに50.504 msのworker窓が一つあり、他の条件の50 ms超過は0だった。
音声一致や既定設定の通過を、遅延設定の資源受入に読み替えない。

未来receiptが既知になるセルは250 msで131、1秒で101となり、先行の静的数え上げと一致した。
ただし実際の未来arrival点予測は両設定とも0だった。`arrival::Engine::integral` は位相係数がゼロでも
`width <= period_seconds / 4` を要求する。今回の周期約0.171744秒では、1秒horizonを16分割しても
この条件を満たさず、定数hazardでありながら62評価後に不明を返した。
独立検証はこの上限到達と不明を確認した。静的なreceipt支持数をheadの点予測成立数へ読み替えない。
次の一単位は、実際に位相依存する場合だけ解像度条件を要求する修正を、非ゼロ位相係数対照とともに検査する。
遅延表の高い再構築費用は別に削減する必要がある。

全体テスト1,173成功・0失敗・38 ignored、31 suite。
`cargo test exit=0 @ 2026-09-18T16:25:55+09:00`。
Clippy、全target check、fmt、release build、diff検査、最終Python検証が通過した。
検証scriptの初回は省略されたresource field、次は別素材向けのfuture residual正例条件で停止し、
対象に合う明示的な独立照合へ修正した。失敗logも保存した。
登録は `i10-evaluation-timing.json`、15取得と入力・binary・検査証拠は
`target/i10-evaluation-timing-20260918/` に保存した。

設定・比較の接続完了と、評価時刻の採用・帰結予測の妥当性・資源受入は別である。
共有/局所共同経路の通常所有、条件付き特徴とscore更新、全head、実身体候補/default消費、
評価時刻の判断、全Voice・worker・live audioの受入は未完。I10全体は未完、I9保留、I11作用無効を維持する。

### 位相に依存しないarrival積分の支持を回復（2026-09-18）

前の比較で見つかった、定数hazardにも周期解像度を要求して点予測を失う問題を修正した。
`arrival::Engine::integral` の `width <= period_seconds / 4` は、contextにpeakがあり、
cosine/sineの係数10/11の少なくとも一方が非ゼロの場合だけ要求する。
厳密なゼロ判定であり、小さい係数をゼロへ丸める閾値は導入しない。
周期長・period support・欠測indicatorの係数は、積分中に一定なので非ゼロでもこの条件を要求しない。
経過時間項は残り、従来の数値収束条件で検査する。

特徴配置、二点Gauss–Legendre、誤差許容 `1e-8 + 1e-6 * abs(sum)`、最大62評価、
reset不明の区間評価、candidate receiptの支持条件は変更していない。
heap確保も追加していない。観測と候補は同じ積分を使うため、観測arrivalとその先のcontinuationも
数値的に変わり得る。観測不変の変更とは扱わない。

Rust検査3件を追加し、arrival関連14件が通過した。
peakなしと三周期、三切片、三経過時間、三horizonの108条件で、周期長・support・欠測係数も含む
定数hazardを独立閉形式と照合した。短周期でも6評価で積分と点予測が得られた。
経過時間に依存するhazardは別の閉形式3条件で検査した。
cosineだけ、sineだけ、両方の非ゼロ対照では、微小な係数も含めて長horizonの解像度条件と62評価上限を維持し、
短窓は別式の10,000点中点積分と照合した。観測と候補の点予測0.5、候補receiptによる新しいgapの解消、
元の観測engineの不変も検査した。元から不明のresetや不完全tailの検査は従来どおり通過した。

通常rendererでは、sustain/stairの三評価時刻と非ゼロ位相係数のstairを前後取得した。
これらには実際のarrival forecastがなかったため、それだけでは実音上の修正確認としなかった。
続いて周期約0.171秒、振幅0.15/0.003の固定パルス素材を追加し、三評価時刻と非ゼロ位相係数の対照を前後取得した。
素材追加は実音上の短周期経路を通す工程確認であり、楽曲や係数の採用ではない。
通常取得は計22回、最大割当取得3回と合わせて25回である。

通常前後の共通観測1,293件を比較した。パルスの三時刻では529件のgroup forecastが不明から0.5へ変わり、
同じ529件のgroup continuationが、正のarrival係数に従う確率分布の移動を示した。
同じgroup・経路mass・他の記述値を維持し、各累積カテゴリ確率が増えないことも検査した。
変更後の通常arrival forecast計1,986件を定数hazardの閉形式で照合した。
候補のreceipt・生存確率・ordinal分布とraw混合も独立計算で照合した。
許容した差はarrivalの確率/評価回数、phrase continuation、その凍結値を使うraw混合だけである。
既存イベント記録・availability・WAVは全条件で一致した。
非ゼロ位相係数の二素材は共通269観測の全非計時値が一致し、パルス対照には実forecast650件と
短周期peakを持つforecast group334件が含まれた。

最大割当は前の比較と同じ両bus・音群1/7・quiet/loud・weightedラベルfalse/trueの16報告行を三時刻で取得した。
後の条件監査で両ラベルともgesture係数はゼロと判明した。非ゼロ係数対照の主張は末尾で訂正する。
最終表の直接投影照合は86,016位置、arrival receiptの独立照合は44,416セルである。
元のfrozen arrival contextと候補receiptは前版と一致した。
未指定の動作時点では発行時40セル/条件、250 ms後では未来131セル/条件、1秒後では未来101セル/条件の
点予測が回復した。三設定16報告行の合計で4,352セルが新たに支持された。
原profileのaccent検出とreset条件を独立に再計算し、支持された全点を0.5・6評価と照合した。
profile範囲外や元のreset不明を救済したわけではない。

処理費用の問題は残る。最大割当の更新最大は未指定41.168 ms、250 ms後168.831 ms、1秒後157.285 ms。
遅延設定はそれぞれ336回の表構築すべてが50 ms超過した。
この比較の前版負荷値は保存済みの取得であり、交互反復した速度差の受入ではない。
通常変更後は15,954 accepted frame・1,716 worker窓、最大60.709 ms、50 ms超過289窓で、
drop/reject/入力欠落は0だった。全worker・Voice・live audioの資源受入は未完了である。

全体テスト1,176成功・0失敗・38 ignored、31 suite。
`cargo test exit=0 @ 2026-09-18T16:40:12+09:00`。
Clippy、全target check、fmt、release build、diff検査、最終Python検証が通過した。
登録は `i10-arrival-phase-resolution.json`、入力・前後binary・25取得・検査証拠は
`target/i10-arrival-phase-resolution-20260918/` に保存した。

今回修正したのは既存arrivalの数値支持であり、評価時刻は採用していない。
実音で確認した観測arrivalと、合成最大割当で確認した候補arrivalは区別する。
通常の実身体に適合した候補点予測やcandidate/default消費、校正・可聴性・作者採用を証明したものではない。
次の一単位は遅延表の再構築費用を測定箇所ごとに分け、必要な帰結を保ったまま重複計算を減らすこととする。
共有/局所共同推論の通常接続、全head、実身体消費、評価時刻の判断、全資源受入も引き続き残る。
I10全体は未完、I9保留、I11作用無効を維持する。

### 動作前の完全hop状態を再利用して遅延表を短縮（2026-09-18）

遅延表の区間計時で、articulation投影が計時した処理の89.5%を占めた。
従来は一つ前の候補の最終prefixだけを保存していたため、動作後の末尾が異なると、
共通する長い動作前区間まで最初から計算していた。
検査用のthread-local区間計時を8箇所に追加した。`cfg(test)` のみで、
ignored assayの `CONCHORDAL_I10_PROFILE_PROJECTION=1` を指定した場合だけ計時する。
通常binaryには入らず、非計時の前後負荷比較では無効にした。
計時対象は特徴窓、grouping、articulation、residual/closure、accent/arrival、continuation、
prefix照合、raw混合であり、全workerの費用内訳ではない。

`ProjectionCache` に最新の完全hop prefixと、動作時刻以前の完全hop checkpointを一つずつ保持する。
次の軌道について各prefixまでの全mask・十一f64値のbit一致を確認し、group・発行時刻・軌道・終端を再検査して、
評価時刻を越えない最も先の状態から再開する。動作後の末尾が違っても、動作前のcheckpointが一致すれば再利用できる。
端数hopは保存せず、遷移・merge・枝刈り・未知質量の計算順序を変えない。
cacheは一つの表構築だけで使い、次の発行時点へ持ち越さない。
heap確保は追加せず、scratchは448から896 byteへ増えた。既存の1,024 byte上限を維持する。

新しいRust検査1件は、同じ前半と異なる末尾、途中から違う軌道、端数hopの動作/評価時刻、
巻戻し、group世代と発行時刻の変更を扱う。直接計算と直列化した全値が一致し、
最初の分岐変更で2 hopだけを再利用すること、異なる前半や新しい発行時刻を再利用しないことを検査した。
既存の部分hop、有限値、未知質量・枝刈り・閉形式を含むprojection関連8検査も通過した。
最大割当では各binary内で最終表の全位置・全Cellをcacheなしの直接投影と照合した。
前後binary間で保存・比較したものは、公開表snapshotと最終arrival記録である。
後者はCell全体の保存ではなく、identity・時刻・arrival・continuation contextの記録と区別する。

負荷比較の条件も訂正した。直近の `i10-evaluation-timing.json` と `i10-arrival-phase-resolution.json` は、
weightedラベルtrueでもstair設定のgesture係数が全ゼロだった。
測定値と元のartifactは保存するが、非ゼロgesture係数の対照を通したという主張を撤回する。
今回の負荷入力には元のsupported-head assayから117個の非ゼロ係数を固定し、
false時だけゼロへ戻した。8つのfalse/true対照すべてで実際のarticulation出力が異なることを確認した。
この係数は工程確認用で、適合・作者採用ではない。

250 ms設定の計時取得では、最大全更新時間は199.389から62.304 ms、
articulationの積算時間は53,577.598から11,569.276 msとなった。
実計算したhopは29,194,368から6,125,952へ約79%減り、再利用分との合計hop数は保存された。
この計時結果と、以下の計時hookを無効にした前後比較は分けて記録する。

非計時の最大割当前後比較は、両bus・音群1/7・quiet/loud・係数ゼロ/非ゼロの16条件を三評価時刻で実施した。
未指定の動作時点は44.130→44.350 ms、50 ms超過0→0。
250 ms設定は198.728→62.501 ms、50 ms超過336→266。
1秒設定は185.285→101.317 ms、50 ms超過336→336だった。
これらは開発用test profile（opt-level 1）の計測値であり、releaseの資源受入には代用しない。
また、当時の集計では `inference_ns` に含まれる `recurrence_ns` を二重加算していた。
元の結果は保存し、重複除去後の値とreleaseでの再測定を末尾「articulation内訳と負荷計測条件の訂正」に記録する。
計時取得を含む64条件対で、1,344組の公開表snapshotと59,904件の最終arrival記録が厳密一致した。
各表のcacheなし直接照合と、実計算/再利用hopの合計保存も通過した。
この負荷試験は合成入力・強制prototype割当であり、実matcher・section・全worker・Voice・audio deviceを含まない。

通常rendererはsustain/stair/pulseの三素材・三評価時刻を前後18回取得した。
共通1,037観測では計時とarticulation計算/再利用counter以外の全観測・候補値が一致し、
既存イベント記録・availability・WAVも一致した。raw head混合は独立計算でも照合した。
変更後12,882 accepted frame・1,386 worker窓の最大は36.311 msで、
50 ms超過・drop・reject・入力欠落はすべて0だった。
これは固定した三素材の結果であり、最大割当や全Voice/live audioの受入を代用しない。

全体テスト1,177成功・0失敗・38 ignored、31 suite。
`cargo test exit=0 @ 2026-09-18T17:04:00+09:00`。
Clippy、全target check、fmt、release build、diff検査、最終Python検証が通過した。
登録は `i10-branch-prefix-reuse.json`、入力・前後binary・26取得・検査証拠は
`target/i10-branch-prefix-reuse-20260918/` に保存した。

次の一単位は、再利用後も区間計時の64.2%を占めるarticulation内部の遷移計算・merge/枝刈りを分けて測定し、
出力と未知質量を保存する削減を検討する。50 msの上限は緩めず、評価時刻の採用も保留を維持する。
共有/局所共同推論の通常接続、全head、実身体候補/default消費、全資源受入も引き続き残る。
I10全体は未完、I9保留、I11作用無効を維持する。

### articulation内訳と負荷計測条件の訂正（2026-09-18）

前節の検査用計時をarticulationの内部へ広げ、遷移・分岐展開、key順sort/merge、
質量順sort/枝刈りを分けた。この三つは既存articulation区間の内数であり、
上位8区間と合わせて足してはならない。`cfg(test)` と明示した計時環境変数の両方を必要とし、
通常binaryの処理や遷移・未知質量・候補の意味は変更していない。

資源測定の条件に二つの問題が見つかった。
まず、直近の最大割当assayは `cargo test --lib` であり、`[profile.dev] opt-level = 1` を使っていた。
通常運転のreleaseはopt-level 3・LTOありなので、開発用の結果だけでreleaseを不合格とする判断を訂正する。
次に、assayの `inference_ns` はrecurrenceの開始から計測していたが、集計時に `recurrence_ns` を再加算していた。
保存済み旧記録の重複しない総時間は `inference_ns + table_ns` である。
現在のassayではrecurrenceの計時終了後に推論の時計を開始し、`timing_schema = "disjoint-v2"` を記録する。
この新形式だけは `recurrence_ns + inference_ns + table_ns` を使う。
旧JSON・旧登録を上書きせず、式とビルド条件を区別する。

同じ入力・117個の非ゼロ係数を持つ対照・8 prototype・16条件を、release設定のtest executableで取得した。
追加の内部計時を無効にした結果は以下のとおり。dev側は前節の取得値から二重加算だけを除去した値である。

| 評価時刻 | dev最大更新時間 | release最大更新時間 | release 50 ms超過 |
| --- | ---: | ---: | ---: |
| 動作時点（既定） | 44.337 ms | 34.649 ms | 0 / 3,216更新 |
| 動作250 ms後の次hop | 62.488 ms | 46.158 ms | 0 / 3,216更新 |
| 動作1秒後の次hop | 101.305 ms | 71.703 ms | 336 / 3,216更新 |

これはコンパイル条件をそろえた確認であり、今回のproductionアルゴリズム高速化ではない。
releaseでもtest専用hookの無効判定は残る。合成入力・強制割当・選択した推論区間の測定であり、
通常worker全体、実matcher、section、queue、Voice、audio deviceの資源受入には代用しない。
250 ms設定はこの限定試験で50 ms以内だったが、評価時刻の作者採用は依然として未決定である。
1秒設定の最大割当はreleaseでも50 msを超える。閾値や候補数を緩めていない。

内部計時を有効にした250 ms設定では、releaseのarticulation積算時間8,252.084 msのうち、
遷移・分岐展開67.38%、sort/merge 21.29%、順位付け・枝刈り7.25%だった。
残りは区間境界・checkpoint等を含む。計時自体の費用が入るため、上表の非計時取得とは分ける。
非線形rate計算を含む分岐展開が依然として最大の費用である。

三評価時刻と内部計時取得を合わせた64条件対で、1,344組の公開表snapshotと
59,904件の最終arrival記録がdev/release間で厳密一致した。
最終表の全Cellは各binary内でcacheなしの直接計算と照合している。
binary間のarrival保存記録はCell全体ではない。この範囲を超えた一致は主張しない。
修正後の計時形式も16条件・336組のsnapshot・15,488件のarrival記録で旧形式との数値一致を検査した。

全Rustテスト1,177成功・0失敗・38 ignored、31 suite。
`cargo test exit=0 @ 2026-09-18T17:27:00+09:00`。
Clippy、全target check、fmt、diff検査、Pythonによる数値照合が通過した。

入力、compiler artifact、binary、計時内訳、照合scriptとログは
`target/i10-articulation-kernel-20260918/` に保存し、`i10-articulation-kernel.json` とfeature manifestへ登録した。
共同推論の通常接続、全head、実身体条件を満たす候補/default消費、全worker/Voice/live audioの資源受入は残る。
I10全体は未完了。I9保留、I11作用無効、既定の動作時点評価を維持する。

### 共同正規化の0音群境界を修正（2026-09-18）

通常の共同推論へ接続する前提を調べたところ、数値核の `Normalizer::normalize` が
音群数0を拒否していた。これは空状態からの起動と全音群退役後の共有文脈更新を妨げる。
§9.1の式では局所項の空積は1、log分配関数の空和は0であり、この場合も
共有親・遷移・共有観測項を通常どおり正規化できる。
音群がない状態と、観測自体の欠測は別である。`observed = false` の全観測項抑制は維持する。

0音群だけの拒否条件を外し、既存の共有正規化・明示unknown・7経路への枝刈りを適用した。
音群を捏造せず、局所列挙数は0、公開する未使用の局所slotは全消去する。
存在しない音群への候補previewは拒否する。共有候補そのものが空、音群数が共有候補間で不一致、
8音群超過、または存在する音群の局所候補列が空の場合は、引き続き拒否する。

独立した60桁Decimal全列挙に4例を追加した。初期unknownだけ、観測あり／欠測の共有更新、
32共有候補の枝刈りを扱う。旧25例は内容不変。
修正前は追加例で `joint group inventory mismatch` により失敗（終了コード101）、修正後は通過した。
Rust検査2件を追加し、0→8→0→1→0音群の更新、既知共有質量とunknownの保存、局所slotの消去、
scratch再確保なし、範囲外preview、不正入力での公開結果保持と復帰を確認した。
共同推論関連20検査が通過した。

これは共同数値核の境界修正であり、通常の観測経路への共同推論接続ではない。
`joint` は引き続き `cfg(test)` 内にある。共有・局所の状態所有、経路ごとの実特徴と共同score、
通常候補表への利用、実身体/default対応、全資源検査は残る。
I10全体は未完了。I9保留・I11作用無効を維持する。

全Rustテスト1,179成功・0失敗・38 ignored、31 suite。
`cargo test exit=0 @ 2026-09-18T17:36:26+09:00`。
Clippy、全target check、fmt、diff検査、29例のDecimal参照再生成一致を確認した。
現在版は既存の `i10-joint-normalization.json` をv2へ更新し、feature manifestに0音群の追補を登録した。
旧登録と旧manifest要約、修正前の失敗、修正後の検査ログは
`target/i10-joint-empty-groups-20260918/` に保存した。

### I10完了境界と通常consumerの照合（2026-09-18、範囲回答待ち）

工程正本 `milestones.md` §2では、I10を「実Voiceの身体・実行結果→私有trace・候補帰結の数値経路→診断」
とし、固定方策、未実行候補の学習除外、両busの対応を検査する。
四つの階層間接続はI12a–d、全処理preflightはR2、実機・通常運用受入はA3に割り当てている。
`implementation-plan-revision.md` §3は「後続全モデルや人の収集を初期実装の完了条件にしない」と定める。
`milestones.md` §6も「後続全階層の完成を現在Iの前提にしない」と定めている。
一方、本台帳の後続記録は共同推論・全head・全資源受入までI10の必須残作業として追加してきた。
これは工程境界の不一致であり、完了しやすさを理由にどちらかへ読み替えない。

ユーザーへ、正本のI10範囲か、共同推論・全head・全資源受入を含む拡張範囲かを質問した。
回答前に追加条件を削除したりI10を完了にしたりしない。I9除外とI11作用無効は選択によらず維持する。
以下はどちらの範囲でも必要な実身体側の確認であり、共同推論の代替ではない。

| 接続 | 現在の実装で確認した境界 | 残る具体的な処理 |
| --- | --- | --- |
| 実行→予測→実音→学習 | `life/self_prediction.rs` のissueは受理済みonset/releaseに限定。観測時にも受理状態・所有世代・元の支持を確認する。`participation_trace.rs` は候補時刻とdefault時刻の私有fit差を診断へ渡す | 未実行候補の予測は、このissue/学習経路を呼ばずに評価する必要がある |
| 実身体の条件付き音響予測 | `runtime/body_profiles/policy_default/energy_projection.rs` が実recipe・残存Tone・予定release・routingから7 classのenergyとdefault差を取得する。呼出元はignored offline assayのみ。`ScheduleRenderer::source_energy_models` もcfg(test)内 | readonlyの条件付き評価を通常経路へ移し、Voice/body世代と両busを保った診断に接続する。未来実音や未来policy入力を予測器へ渡さない |
| 通常body default | `action_candidates/live.rs` は20 Hz以下で最初の受理指令をclass化し、なければ活動中continue／予約onsetを記録する。複数受理数は保持するが、default_inputは指令列全体ではない | 第一指令だけを全defaultの帰結と扱わない。既存Tone・確定予約を含む評価対象を定義し、不明な将来入力と既知ゼロを分ける |
| 共有候補表 | `temporal_cognition/observation.rs` が表を構築してsnapshotへ渡す。現在の`life/`にはこの共有表を読むconsumerがない | 身体条件・bus・世代・支持時刻・期限・candidate/default双方の支持を検査して、差分を通常診断へ渡す。descriptor近傍だけを身体適合の証明にしない |

次の実装単位は、実身体の条件付きcandidate/default評価を通常のreadonly診断へ接続すること。
評価しただけの候補を指令として発行せず、予測pendingや私有学習を増やさない。
既存の実身体offline比較を同じ核へ通し、固定方策の音声不変性、候補による学習不変性、
複数指令・予約・routing・世代変更・未知支持を検査する。全7 class、既存の比較失敗、
未接続の帰結項は保持し、一つのenergy差だけで全候補帰結を完成としない。

この照合では実装を変更していない。現在の最新全体検査は前節の1,179件成功を参照し、
今回新たに実機・全負荷・通常candidate/default接続が通ったという主張はしない。

### 実身体の条件付き候補energyを通常診断へ接続（2026-09-18）

`action_candidates::energy` に、実rendererの発行時recipeと残存Toneから条件付きenergyを計算する
readonly経路を追加した。以前のoffline取得器も同じ16点の物理窓積分を呼ぶ。
既存20 Hz以下の身体診断が標本化した、単一の受理済みonsetを入口とする。
source／body世代、routing、envelope、control、sine位相、既知の予約releaseをコピーし、
64入力buffer・64出力recordの別workerへ渡す。renderer上の残存source走査は64件までとし、
複数指令や予約onsetを含む未対応入力は評価対象に混ぜず、unsupportedへ数える。

比較する7 classは既存の `Class::input` に従う。既定の動作時点評価を維持し、時刻候補は
内在周期2個分または4秒までの既存gridを使う。条件は「この発音recipeと既知release、
凍結したcontrol、以後の追加policy指令なし」である。defaultは同じ局所onsetであり、
未来policy全体のdefault予測ではない。未発行の候補はcommand、予測pending、私有学習へ入れない。

通常reportの `body_candidate_energy` は両busの250 ms／1 s／2 s／4 s窓について、
平均energy、default平均、差分、16点の支持数とcoherent支持数を記録する。
UIはworkerの投入・完了・未対応・取り落としを表示する。未知支持をゼロへ置き換えず、
非routingまたは発音不在が確定した区間だけを既知ゼロとする。全積分点はoffline記録に残す。

検証結果:

- 通常6素材で9候補record、7 class、両bus2,792窓を発行時receiptへ照合した。
  6組のWAV、既存のcommand／帰結／予測学習record、終端observerの既存値、共通566時間観測は一致した。
  対象外2件、入力・出力の取り落とし0件、worker失敗0件だった。無reportとobserver-offの対照も音声一致した。
- offline71分岐・両bus9,088積分点／568窓は独立Python参照で検証した。
  非sineの3素材は共有化前と完全一致した。sineの404数値座標には最大2.17e-19の差があった。
  構造・支持・入力は一致し、順位判定も一致した。既存の予測誤差・順位反転は解消した扱いにしない。
- 実rendererの追加検査では候補診断の有無による音声・実帰結・予測学習の一致と複数指令の拒否を確認した。
  窓の既知ゼロ／未知、releaseの適用対象、容量上限、worker終了時の回収も検査した。
  全Rustテスト1,182成功・失敗0・38ignore、Clippy、全target check、fmt、release build、diff検査を通過した。

取得と検証は `target/i10-live-candidate-energy-20260918/`、登録は
`i10-live-candidate-energy.json` とfeature manifestの `i10_live_candidate_energy` に保存した。
この素材の背景計算最大2,866 µsは全worker／全Voice／実機の資源受入を意味しない。

次は、共有候補表を読む通常consumerで、実身体への適合、bus・世代・支持時刻・期限、
candidate/default双方の支持を検査し、私有traceの差分と対応させる必要がある。
複数指令・未予約機会、energy以外の帰結、実身体予測精度も残る。
完了範囲の回答前に共同推論・全head・全資源受入の追加条件を取り消さない。
I10全体は未完了。I9保留、I11作用無効、既定の動作時点評価を維持する。

### 実身体候補と発行時私有traceの接続・offline観測同期（2026-09-18）

実身体energyの候補packetに、受理済みonsetの発行時referenceとfull／elapsed-only両head分布を
凍結して渡す経路を追加した。command ID、source／世代、epoch、bus、発行・inventory時刻を
reportへ残す。私有Bankの現在値を背景workerから読み直さず、既存の64 packet内に最大16 referenceを
保持する。Frozenは20,000 byte以下を検査した。これは既存容量に加わる固定領域であり、資源受入は別途必要である。

onset_now／delayed_onsetだけを、既存の「onset + 1 sample」という固定時刻モデルへ写し、
同一defaultと同じ支持anchor質量で比較する。これは発音時刻の校正ではない。
空head・未学習headは未知のままとし、presentation-only、release／gap／continue／skip／waitへ
onset分布を流用しない。候補はcommand発行、予測pending、信用更新へ入らない。
世代交代やその後のcontext更新でも凍結値が変わらず、別世代に貸し出せないことを検査した。

通常接続の検証で、deterministicなoffline実行でもTapへの送信完了後に共有観測workerが残り、
reference contextの遅延によって私有traceの鮮度条件を満たさない問題を確認した。
初回のsine-flow比較では2件のinventory時刻／entryが実行間で変動し、学習数は0だった。
0.1秒の鮮度条件を緩めず、offline Tapはsnapshotとreference contextの公開完了を待ってから
analysisへ戻るよう修正した。live Tapの非ブロッキング32件queueは維持する。
公開前の待機、拒否frameの応答、終了処理と既存のqueue欠落検査を通過した。

同期修正と候補診断の影響は二つの比較へ分けた。

- 同期前後の6素材ではWAV・実行指令・既存自己音予測は一致した。
  鮮度不足の解消によってharmonic-entrainedの私有学習数は0→2、sine-flowは0→21になった。
  これは同期修正の効果であり、候補を学習させた結果ではない。
- 同じ同期修正を含み、候補へのtrace添付だけを外した対照binaryと比較した。
  WAV6組、既存energy、実帰結、私有trace記録、終端学習カウンタ、共通696時間観測は一致した。
  9件のenergy recordに含まれる100組のfull／elapsed-only候補差分を、reportの発行時分布から
  独立Fraction積分で再計算した。支持あり22組、非ゼロ差分20組、最大誤差4.61e-19だった。
  別headの152候補entryは未対応として保持した。

全Rustテスト1,184成功・失敗0・38ignore、Clippy、全target check、fmt、release buildを通過した。
対照ビルドが共有Cargo出力先を上書きした取得も一度発生したが、binary hashと新report項目の欠落で検出した。
当該取得は不採用とし、通常版を再ビルドして異なるbinaryと新項目を必須検査した取得を正本にした。
不採用の取得・初期同期不一致・テスト失敗も保存し、成功扱いにはしていない。

証拠は `target/i10-body-trace-candidates-20260918/`、採用した通常比較はその `synchronized/`。
登録は `i10-body-trace-candidates.json` とfeature manifestの `i10_body_trace_candidates`。
共有候補表は依然として要約の公開に留まり、通常consumer、実身体への適格性、group世代／期限／
candidate-default支持の検査は未接続である。別headの事象モデル、非energy帰結、全資源受入も残る。
I10全体は未完了。完了範囲の回答待ち、I9保留、I11作用無効、既定の動作時点評価を維持する。


### 共有候補表の読取専用consumerと実音での拒否検査（2026-09-18）

共有workerが構築した表から、両raw headと出所を保持する小型の不変publicationを公開する経路を追加した。
通常runtimeはsnapshotと同じkeyの表だけを非ブロッキングで取得し、受理済みonsetの既存packetへ
Arcと身体descriptorの対応を凍結する。背景workerは新しい表を読み直さず、そのpacketだけで
候補と同じ局所defaultを比較する。表の失効・EOFは公開先へ反映する。
候補は実行指令・予測pending・学習へ入らず、I11の生成作用も追加しない。

表は7 class×8 prototype×32時刻の小型セルを保持する。通常reportと単体検査で1 MiB未満、
再構築検査では元の全Cell表の4分の1未満を確認した。これは追加領域であり、全体資源受入ではない。
source／身体世代、bus、model、支持の可用時刻、0.5秒未満の鮮度、groupのbus／epochを検査する。
未接続busのdefaultは専用の拒否理由を返す。整数化済み時刻gridの最近点を選び、同距離なら前を選ぶ。
選んだセルが未知なら隣へ探索せず、補間もしない。candidate／defaultのraw分布・支持・出所を保持し、
両側に支持のあるheadだけexpected ratingの差を返す。descriptor対応は身体転用の適格性とは扱わない。

セル選択の31境界とtie、欠測セル、片側head欠測、支持ゼロ、所有者／世代／bus／modelの不一致、
未来支持、期限切れ、group交代を単体検査した。表の再構築後も、発行済みArcから読める値が
変わらないことを既存の評価時刻再構築検査で確認した。

通常rendererの変更前後7素材では、WAV7組、実行指令・身体default・帰結・私有trace・既存energy、
終端学習カウンタと共通858時間観測が一致した。14件の候補recordに両bus計28表が到達し、
全28件の出所を通常観測reportと照合した。候補bus entryは身体対応なし994件、未接続bus194件。
**通常実音のpaired正例は0件であり、帰結差分の実用経路が成立したとは判定しない。**

816件の身体descriptor距離を独立に再計算した。候補発行時の発音対象bus24件はすべて固定距離閾値0.25の外で、
残った4件の対応は無音bus側だった。110 Hz／amp 0.04の追加flow素材でも、静的medoidと同じ周波数・振幅だけでは
対応しなかった。閾値・尺度・profileは変更していない。この結果は実身体への適格性を近傍だけで扱えないという
既存監査と整合するが、新しい適格性モデルを検証した結果ではない。

初回の検証器では、hop発行のbody_defaultと、許可済みonsetを原点にしたenergy実験のdefaultを
完全同一Inputと誤って比較した。同一tone・実機会時刻・excitation時刻の照合へ訂正し、
既存energyと私有traceの変更前後完全一致検査は維持した。
全Rustテスト1,186成功・失敗0・38ignore、Clippy、全target check、fmt、release buildを通過した。
証拠は `target/i10-shared-consumer-20260918/`、登録は `i10-shared-consumer.json` と
feature manifestの `i10_shared_consumer` に保存した。

次に必要なのは、実身体と実defaultに条件づけた帰結を、未知を保ったまま通常経路で正例まで示すこと。
現在の接続は単一の許可済みonset recipeと既存Tone・既知releaseに限定され、未予約機会や複数指令のdefault、
別headの事象モデルと非energy帰結は未完である。共同推論・全head・全資源受入と完了範囲の未回答も残る。
I10全体は未完了。I9保留・I11作用無効・既定の動作時点評価を維持する。


### 継続中の実身体候補と共有表の通常正例（2026-09-18）

受理済みonsetだけでなく、継続中の既存身体からcontinue／release／gapを評価する通常経路を追加した。
既存のsource別20 Hz以下のsample時に、同じ身体世代の活動中Toneを最大64個まで既存packetへ凍結する。
同hopに指令・新recipeがある場合、予約onset・別source世代・不明な身体世代がある場合はこの経路を使わない。
元のTone、時計、指令列は変更せず、新しいonset用の私有traceや予測pendingも作らない。

defaultは、発行時に存在するToneと既知releaseをそのまま継続した条件付き予測とする。
未来のpolicy指令全体ではない。releaseは候補時点で活動中のToneだけに適用し、自然終端・既知終端後は除外する。
gapの期間は実発音receiptと同じ四捨五入による整数sampleへ写す。周期がない場合はgapを作らない。
今回の無追加発音モデルでgapが予測する音響差は既存Toneのrelease部分であり、後続発音を抑えた効果の実証ではない。

250 ms／1 s／2 s／4 sの窓には既存の16点energy・coherent sine核を使う。
候補とdefaultの双方が既知の場合だけ差を返し、未知をゼロに置き換えない。
新recordは `tone_id: null` とし、身体全体へ架空の代表tone IDを付けない。
`default_routed` は既存Toneの実routingから求め、発行時source／身体世代とともに出力する。
既存onset recordのtone IDは維持し、raw共有表の読取専用契約も共用する。

既知releaseのある定常energyを手計算で照合し、未知bus、release後の既知ゼロ、候補差分、
周期欠測、終端後のrelease除外、予約Tone・空身体・容量超過の拒否と背景workerの処理を検査した。
全Rustテスト1,187成功・失敗0・38ignore、Clippy、全target check、fmt、release buildを通過した。

通常8素材で新規651件の継続身体record、87,656物理窓を取得した。48,792窓で非ゼロenergy差を確認した。
身体defaultとの同一source／世代・routing・活動数・指令なしの対応、候補入力、窓とdefault差を独立に照合した。
WAV8組、既存の14件のonset候補record（共有表の出所・結果を含む）、実帰結・私有trace・学習カウンタ、
共通910時間観測は変更前と一致した。候補workerの処理件数・費用は増加したため、同一性判定から分離した。
取得した665件は全件reportへ到達し、capacity／output欠落・worker未対応・worker失敗は0だった。
背景処理最大は2.826 ms。これを全Voice・全worker・audio hopの資源受入とは扱わない。

追加した静的110 Hz素材は、以前登録したprototype取得条件の再利用である。
共有表から64組（continue32・release32）のcandidate／defaultが返り、両busの分布・支持・出所・整数grid選択を照合した。
これは通常consumerの到達正例であり、独立素材への身体転用の検証ではない。
closureは全64組で支持なし、continuationは支持ありだが候補差は全件ゼロだった。
既存の250 ms後評価も比較条件として固定して取得したが、同じ64組で差はゼロのままだった。
**非ゼロordinal帰結の通常正例は未証明。** 評価時刻の既定値・prototype・距離閾値・尺度・係数は変更していない。
reportを付けない静的110 Hz取得でも音声は完全一致した。

証拠は `target/i10-active-body-candidates-20260918/`、250 ms比較はその `timing250/`。
登録は `i10-active-body-candidates.json` とfeature manifestの `i10_active_body_candidates`。
共有表の到達と実身体energyの非ゼロ差分は接続したが、身体転用の適格性、非energy帰結、
未予約機会・複数指令を含むdefault、別headの事象対応、全資源受入は残る。
完了範囲の未回答と共同推論・全headの未完も維持する。I10全体は未完了、I9保留、I11作用無効。


### 同一候補窓の厳密な再利用と負荷比較（2026-09-18）

一つの凍結済みrequest内で、追加発音もreleaseもない候補の物理窓を共用するよう変更した。
release／gapは同じrelease時刻の窓だけを共用する。発音追加は別扱いとし、request間のcacheは作らない。
候補のInput、default差、支持、onset trace、共有headを混ぜず、既存の有界な時刻配列と同じ16枠へ計算結果を保持する。
これは同一入力の重複計算除去であり、窓・点数・積分・モデルの近似変更ではない。

予約releaseあり／なし、未知bus、自然終端後のgapを含む候補を、再利用なしの直接projectionと照合した。
通常8素材のWAV、665件の全候補記録（計時値を除く）、既存学習・帰結、共通910時間観測は完全一致した。
候補記録の物理入力keyから算出した窓計算は92,408回から49,672回へ減少する。
この回数はruntime計器による実測ではなく、検証済みの全候補とreuse keyに基づく計数である。

sine-flowと110 Hz-flowを同じrelease binaryの変更前後で交互3往復した。
各取得で候補記録と音声の一致を再確認し、6比較すべてで背景処理の合計が34.4〜48.5%減少した。
sine-flowの最大は変更前2.861〜3.047 ms、変更後1.440〜1.747 msだった。
1.6 ms以内の保証は未成立であり、全Voice・全worker・実機の受入でもない。

全Rustテスト1,188成功・失敗0・38ignore、Clippy、全target check、fmt、release buildを通過した。
証拠は `target/i10-energy-reuse-20260918/`、登録は `i10-energy-reuse.json` とfeature manifestの `i10_energy_reuse`。
身体適合・非energy帰結・full default・共同推論等の未完と、I9保留・I11作用無効を維持する。


### 同一時刻のenergy二重評価除去（2026-09-18）

各Tone／時刻について、通常energy加算とcoherent sine成分生成が同じ `ToneEnergy::at` を二度呼んでいた。
候補窓内では一度求めた値を両方へ渡すよう変更した。既存のsine予測入口は同じ計算を行うwrapperとして維持し、
未知、既知ゼロ、位相と支持条件は変更しない。ここでの対照は、直前の窓再利用をすでに含む版である。

通常8素材のWAV、665件の全候補記録、既存学習・実帰結、共通910時間観測は完全一致した。
全Rustテスト1,188成功・失敗0・38ignore、Clippy、全target check、fmt、release buildを通過した。
今回の物理窓数は対照と同じ49,672であり、窓数減少を二重に数えない。

事前に「数値が一致し、同じ二素材の交互3往復すべてで背景処理合計が減る場合だけ残す」と固定した。
6比較すべてで合計費用が5.7〜34.5%減少したため、この変更を保持した。
一方、sine-flowの一比較では最大値が1.440→1.676 msへ増加した。この最大値の退行は保存し、
全指標の改善や1.6 ms以内の成立とは判定しない。他のsine-flow二比較の変更後最大は1.277／1.272 ms、
110 Hz-flow三比較の変更後最大は0.518／0.516／0.479 msだった。
全Voice・全worker・live audioの受入は引き続き未完である。

証拠は `target/i10-energy-point-reuse-20260918/`、登録は `i10-energy-point-reuse.json` と
feature manifestの `i10_energy_point_reuse`。直前の窓再利用は別登録・別binary・別比較として保持した。
身体適合、非energy帰結、複数指令等のdefault、私有traceの別head対応、共同推論と完了範囲の未回答は残る。
I10全体は未完了。I9保留・I11作用無効・既定の動作時点評価を維持する。


### 4／16／64 Voiceの短期負荷screen（2026-09-18）

既存のwindow／point再利用を含む版で、hold／flow各4・16・64 Voice、6秒＋release後0.5秒を取得した。
seed・設定・全source・入力・binaryを事前固定し、各条件でrendererのobserve／off音声と、
`profile-alloc`付きinstrumentのreportあり／なしを分けた。計24実行はすべてexit 0。
instrumentはデバイスなし・音声保存なしであり、WAVは別のrendererだけが保存した。
6組のWAV、onset、Population記録はobserve／offで完全一致した。

| 条件 | 全Voice稼働hop p99、reportなし | reportあり | 8.5333 ms短期判定 |
| --- | ---: | ---: | --- |
| hold 4 | 2.3515 ms | 2.5800 ms | 両方以内 |
| hold 16 | 2.9578 ms | 3.1390 ms | 両方以内 |
| hold 64 | 5.7943 ms | 5.8255 ms | 両方以内 |
| flow 4 | 2.7470 ms | 2.8917 ms | 両方以内 |
| flow 16 | 5.4412 ms | 5.6554 ms | 両方以内 |
| flow 64 | 10.2485 ms | 11.3673 ms | 両方超過 |

値は個々の実hop全体のp99であり、phase別p99を加算した値ではない。
flow 64のsynthesis区間p99はreportなし7.4722 ms、あり6.3814 msだった。
この区間には実合成だけでなく、指令適用・私有観測・自声照合・背景worker待機も入る。
内訳計測前に音源計算だけの問題とは判定しない。

reportありinstrumentで候補は合計18,289件を提出・完了し、capacity／output drop・worker失敗は0。
64 Voiceで最大64件／発行hop、半開100 ms区間内128件となった。
共有workerは両bus・全条件で50 ms超過と入力dropが0。100 ms窓p99のhistogram上限は16 ms、
実構築の対応prototypeは最大3個であり、全8個の負荷ではない。
私有bodyは全条件611 frame、capture drop・invalid hopは0。ただし全frame／窓のp99は未計測。
reportなしの背景worker終端計器と候補queue待ち・消費時鮮度も未取得である。

**計時基準の訂正:** 直前二節の1.6 ms言及は個別requestの最大値を比較したものだった。
正本の1.6 msは、全個体の新規判断を合計した一hopのp99に対する条件である。
個別背景requestの最大値や、発行hop別にまとめ直した背景費用から、この合否は判定できない。
過去の実測と登録は変更せず、この区別を今後の判定に適用する。

証拠は `target/i10-population-preflight-20260918/`、登録は `i10-population-preflight-screen.json` と
feature manifestの `i10_population_preflight_screen`。既存計器の検査5件は成功した。source変更はない。
60秒warmup＋600秒計測、全候補負荷、全処理・実機受入の代替にはしない。
64 Voice flowの超過原因を次に切り分ける。I10未完了、I9保留・I11作用無効を維持する。


### renderer内訳の通常profile接続と再現性の切り分け（2026-09-18）

`--profile`時だけScheduleRenderer内のsetup・指令適用・sample処理・自声history・私有観測完了・
PCM提出を6区間で計時するよう追加した。profileはschema 3とし、各値は`synthesis_us`の内数である。
全hop・外側の合成計時を維持し、合成に含まれる照合reportやoffline待機を隠さない。
通常の音声・方策・候補・学習式は変更していない。追加のhop heap確保はない。
既存の評価CLIもschema 3の内訳検査・集計へ接続し、比較元の保存schema 2を読める経路を維持した。

全Rustテスト1,188成功・失敗0・38ignore、Clippy・全target check・fmt・release buildが通過した。
profile有無に関わる5件のintegration検査と、評価CLIのPython 28件も通過した。
初回全体compileではfixture二箇所の新field不足を検出し、補完してから全体検査をやり直した。
その失敗logも保存した。OSの`perf_event_paranoid=4`によりperf取得は不可だったため、
OS設定を変更せず、この内訳計器を使った。

hold／flow各4・16・64 Voiceについて、instrumentのreportあり／なしとrendererの18実行を追加した。
6組のWAVは前節の凍結binaryと完全一致。onset・Population・body default・私有trace記録も一致した。
候補recordの差は共有bindingの`available`のみで、energy・私有trace・共有headの数値差はなかった。

64 Voice flowの全hop p99はreportなし10.2029 ms、あり11.9292 msで、8.5333 ms超過を再現した。
reportなしの6区間の合計wall時間に対しsample処理は約70%を占めた。sample区間p99は5.3259 ms、
自声historyは1.6226 ms、私有観測完了は0.6190 ms、指令適用は0.1710 msだった。
これらのp99は加算しない。sample区間にはTone音源、sample単位の実帰結観測、両bus加算、
自声／私有PCM captureを含むため、この結果だけで音源演算単独を原因としない。

厳密な全診断一致は不合格として保持する。hold 64とflow 16／64の一部でdescriptorの
到着時刻・参照支持・候補bindingの時刻が変わった。flow 64 rendererではdescriptor予測2,218座標、
発行時更新回数4箇所の差に加え、通常自声予測・誤差集計にも差が出た。
私有body workerは4 bufferの返却を待つだけで、各frameのsnapshot公開完了をoffline生成側へ
同期していない。`available`も並行して進む生成clockを読む。この構造は時刻・学習順序の変動を許す。
計時変更だけの影響と決めつけず、変更前の同一binaryを再実行する対照を追加した。
この対照でも音声は完全一致し、descriptorの到着時刻90箇所、発行時の参照支持20箇所、
候補bindingの到着時刻4箇所に差が出た。ただし、この一回の同一binary対照では学習数値の差はなかった。
到着変動が旧版にも存在することは確認したが、前後比較の全数値差の因果的な帰属はまだ完了していない。

証拠は `target/i10-render-phase-profile-20260918/`、登録は `i10-render-phase-profile.json` と
feature manifestの `i10_render_phase_profile`。次の対象は私有bodyのoffline受け渡し順序であり、
固定方策の音声、実際の公開時刻、学習順序を分けて検査する。liveの非同期経路をoffline同期へ置換しない。
今回の計時は全資源受入でも、全診断の再現性成立でもない。I10未完了、I9保留・I11作用無効を維持する。


### 私有bodyのoffline公開順序を確定（2026-09-18）

私有bodyのdeterministic経路に、PCM frameごとの公開完了通知を追加した。
workerはdescriptor／prototypeを同じsnapshotへ公開し、bufferを返してから通知する。
生成側は`Capture::end`でこの通知を受け取ってからsnapshotを参照し、次の生成clockへ進む。
descriptorを更新しないframeにも完了通知を返す。live経路には通知channelを作らず、
既存の4 buffer・非待機提出・欠落計数を維持する。身体値・学習式・方策・routingは変更しない。

snapshot lockを検査側が保持し、offlineの`end`が公開完了まで戻らないこと、liveの`end`は
同じlockが保持されても戻ることを検査した。descriptor非更新frame、両busの最初の記録、
buffer全数回収、EOFも確認した。身体関連11件と、全Rustテスト1,189成功・失敗0・38ignore、
Clippy・全target check・fmt・release buildを通過した。

前節と同じhold／flow各4・16・64 Voiceを用い、修正後rendererを各2回、instrumentを
reportあり／なしで各1回、計24実行した。すべてexit 0で、各body終端は611 frame、
capture drop・invalid hopは0。修正後の全descriptorで`available == end`を確認した。
これはoffline生成clockを公開完了まで進めない結果であり、実機の分析遅延をゼロとする変更ではない。

変更前後の6組と修正後反復の6組でWAVは完全一致した。onset・Population・body default・
私有trace、descriptorの物理値・元の支持区間も前後一致した。候補energyと私有traceの数値も不変。
一方、旧版の非同期参照時点からは、body binding・支持状態と一部の予測／学習値が変わった。
この差を計時値と一緒に捨てず、`before-after-differences.json`へ保存した。
候補の前後差は共有body bindingと共有照合statusに限定され、身体適合の改善とは判定しない。

修正後rendererの2回とinstrument report実行では、計時値と非同期の出力順を除いた診断内容が全6条件で一致した。
一回分の対象は、候補18,289件、身体descriptor19,824件、実帰結2,490件、descriptor予測2,094件、
私有trace1,541件と各終端学習集計。両busの共通756観測も一致した。
候補18,289件は提出数＝完了数＝記録数で、capacity／output drop・worker失敗は0。
未対応数はflow 4／16／64で1／2／3件のまま保持した。未実行候補を学習へ回す変更はない。

同期後の64 Voice flow全hop p99はreportなし15.1642 ms、あり16.8990 msだった。
`capture_delivery_us`にはofflineの私有worker完了待ちが入り、そのp99はそれぞれ7.0031／5.3711 ms。
短期wall時間の8.5333 ms条件は不合格のまま保持する。今回の同期はoffline再現性の修正であり、
CPU費用削減・live deadline成立・全資源受入の証拠ではない。

証拠は `target/i10-body-publication-order-20260918/`、登録は `i10-body-publication-order.json` と
feature manifestの `i10_body_publication_order`。過去の非同期入力に基づく取得・失敗・登録は保存した。
全窓の私有worker費用、reportなしの背景終端計器、候補queue／鮮度、全8 prototype・長時間・実機検査は残る。
身体転用・非energy帰結・full default等も未完。I10未完了、I9保留・I11作用無効・既定の動作時点評価を維持する。


### 私有worker全frame計器とreport非依存の終端状態（2026-09-18）

共有workerの有界resource meterを`temporal_cognition/resources.rs`へ移し、私有bodyでも共用した。
全受信frameと100 ms窓を、descriptor公開の有無によらず計数する。40 msのhistogram境界を追加し、
50 ms上限と20% headroomを区別した。私有frame計時は解析・prototype・snapshot公開lock／copyを含むwall時間であり、
CPU時間ではない。deliveryはqueue待ちも含む。meter更新とbuffer返却・完了通知はframe計時の外にある。

instrument profileをschema 4とし、drain／join後の両bus共有worker・私有body・候補energyの終端状態を
reportなしでも取得するよう接続した。無効・未計測はnullで保持する。終端分布は全runの集計であり、
指定したwarmup除外窓の集計へ読み替えない。評価CLIはschema 4を読み、旧schema 2／3は欠けた計器を補完しない。

全Rustテスト1,189成功・失敗0・38ignore、Python評価CLI29件、Clippy・全target check・fmtが通過した。
私有全frame数と窓費用の保存、EOF、40 ms境界、report有無での終端取得、reportとの同じbody終端値を検査した。
初回compileで旧fixtureの追加field不足を検出して補完し、全体検査を実施した。

release版でhold／flow各4・16・64 Voiceを、rendererとinstrumentのreportあり／なしで各実行し、
18実行すべてexit 0となった。前節の公開順序修正版と6組のWAVが完全一致した。
計時と非同期出力順を除く9種類の診断内容も前版およびinstrumentと一致し、
候補18,289件・身体descriptor19,824件・実帰結2,490件・descriptor予測2,094件・
私有trace1,541件、各比較の両bus共通756観測を確認した。

全条件の私有bodyで611 frame、66窓（完全配送65窓）を回収し、histogramの件数と
全frame費用＝全窓費用の保存を検査した。reportなしでも両bus共有worker・私有body・
候補energyの終端状態を取得できた。capture／候補capacity／output drop、共有欠落は0。
候補提出＝完了＝記録数を維持した。flowの既存未対応数1／2／3件は欠落とは別に保持する。

64 Voiceの私有workerは、holdとflowのreport有無すべてで100 ms窓のp99が
50 ms超〜75 ms以下のhistogram区間に入った。40 msのwall時間screenは不合格。
50 ms超の窓数はholdで24／27、flowで25／33（reportあり／なし）、
全窓最大は順に55.244／57.711、54.816／60.261 msだった。
4・16 Voiceの私有窓最大は25 ms以内、共有workerの全条件・両busの窓最大は16 ms以内だった。
ただし共有側の対応prototypeはこの入力では最大3個であり、全8個の受入ではない。
私有deliveryの25 ms超はhold 4のreportなしとflow 4のreportありに各1件あり、これも保持した。

64 Voiceの全hop p99はholdで10.9278／11.0164 ms、flowで18.0312／17.2421 ms
（reportあり／なし）で、8.5333 msを超過した。offlineの私有worker完了待ちを含むwall時間であり、
CPU費用やlive deadlineへ読み替えない。histogramのp99はnearest-rankの区間であり、
全runの66窓と完全配送65窓はそれぞれ独立に保存した。warmup除外後の分布ではない。

証拠は`target/i10-background-resources-20260918/`、登録は`i10-background-resources.json`と
feature manifestの`i10_background_resources`。計器と非干渉の検査は完了したが、負荷screenは不合格。
次は私有bodyの解析費用と、残る候補帰結／defaultの通常接続を扱う。候補queue／鮮度・判断費用、
全8 prototype・正式な長時間・実機受入は未検証。I10未完了、I9保留・I11作用無効。

### 私有解析にも使うNSGTのFFT作業領域を再利用（2026-09-18）

`RtNsgtKernelLog2`は毎hopでRustFFTの`process`を呼び、内部で作業用Vecを確保していた。
解析器の構築時に必要長の作業領域を確保し、`process_with_scratch`へ渡す形へ変更した。
FFT・kernel・平滑化・特徴量の計算順序、身体・方策・routing・既定評価時刻は変更しない。
各解析器は必要長のComplex32配列を常時保持し、cloneでも複製する。全体のメモリ受入は別途必要である。

`profile-alloc`付きの回帰検査は修正前に14確保・28,672 bytesを検出して失敗し、修正後は0となった。
coherent／incoherent、FFT長256／2048、最初の処理・reset後・clone後、空／短い／複数hopと
callback経路を検査した。これは当該NSGT処理の確保数であり、body worker全体の無確保を意味しない。
NSGT関連6検査、全Rust 1,189成功・失敗0・38ignore、Clippy・全target check・fmt・release buildが通過した。

前節と同じ6条件の18実行でWAV、計時・出力順を除く診断内容、共通756観測が前版と一致した。
候補18,289件、身体descriptor19,824件、実帰結2,490件、descriptor予測2,094件、私有trace1,541件を照合した。
続いてhold／flowの64 Voiceを旧版・新版で各3往復、計12実行した。実行順は往復ごとに反転した。
私有frameの合計wall費用はholdで6.6〜7.7%、flowで5.7〜8.9%減少した。
一方、全hop p99が増えた対もあり、64 Voiceの40 ms窓headroomと8.5333 ms全hop条件は未達である。
反復中の新版でも50 ms超の私有窓がholdで0／2／2件、flowで2／10／7件あり、失敗を保持した。

証拠は`target/i10-nsgt-scratch-20260918/`、登録は`i10-nsgt-scratch.json`と
feature manifestの`i10_nsgt_scratch`。次の費用調査対象は、128 laneで複製している不変NSGT kernelである。
候補帰結・full default・身体転用の通常接続と検証も引き続き残る。I10未完了、I9保留・I11作用無効。

### NSGTの不変kernel共有と独立した解析状態（2026-09-18）

`RtNsgtKernelLog2`のcloneが帯域kernelと周波数gridを丸ごと複製していたため、
不変の`NsgtKernelLog2`だけをArcで共有した。ring、FFT buffer／scratch、平滑化と出力は各cloneが所有する。
48 kHz・FFT 2048・hop 512・55〜8000 Hz／96 bins per octaveで128個を複製した際の確保は、
両power modeとも89,344回・124,754,944 bytesから640回・6,302,720 bytesへ減った。
これは元kernelの構築と外側の予約済みVecを除く確保要求量であり、process RSSではない。

確保量の回帰検査は変更前に失敗し、変更後に通過した。履歴を持つcloneへ異なる入力を与え、
片側だけresetし、元の解析器を破棄した後まで、独立に構築した参照とbit単位で照合した。
NSGTのfeature付き8検査、全Rust 1,190成功・失敗0・38ignore、Clippy・全target check・fmt・release buildが通過した。
6条件18実行のWAVと計時・出力順を除く診断内容も一致した。候補18,289件、身体descriptor19,824件、
実帰結2,490件、descriptor予測2,094件、私有trace1,541件、共通756観測は不変だった。

速度改善は成立しなかった。64 Voiceを旧版・新版で各3往復した12実行では、私有frame合計wall費用が
holdで2.4〜5.0%、flowで1.5〜3.4%増加した。holdの50 ms超の私有窓は旧版の0／0／0件から
新版の2／6／12件へ増え、flowも9／1／3件から16／15／16件へ増えた。
全hop p99はholdで増加、flowで低下したが、いずれも8.5333 msを超えた。
約118.5 MBの複製確保を削減する変更として保持し、計時の退行と未達の40 ms条件を未解決として残す。
この短期wall時間比較からCPU時間・実機・全資源受入を認定しない。

証拠は`target/i10-nsgt-shared-kernel-20260918/`、登録は`i10-nsgt-shared-kernel.json`と
feature manifestの`i10_nsgt_shared_kernel`。次はfull defaultの通常接続へ戻る。
現状は第一受理指令の診断と単一onset／既存bodyの条件付きenergyであり、複数指令・予約onsetを含む
default全体の帰結ではない。未発行の将来方策入力を予測器へ混入させず、この差を埋める。
I10未完了、I9保留・I11作用無効を維持する。

### 複合指令と予約onsetを含む既知defaultの候補投影（2026-09-18）

従来は単一On指令の発行時だけに限定していた候補packetを、そのsampled batchの全On／Offと
Update予約の適用後に固定する経路へ変更した。既存の20 Hz標本化で選ばれた一つの発音recipeを焦点とし、
それ以外の受理済みTone・予約onset・既知releaseと各routingを、基準と候補の双方へ保持する。
実際の先頭受理指令は`default_schedule.first_transition`、焦点機会は従来の`default_input`へ別々に記録する。
Off→Onや複数Onでも、一source・一batchにつき候補packetは一つで、64 Tone／64 packetの上限を維持する。
現在発音中の身体を扱う既存経路も、予約Toneを一律拒否せず同じ数値核へ渡す。

gapは`[action_at, withhold_until)`にonsetを持つ予約Toneを抑止し、上端時刻のonsetと既存のrelease tailを保持する。
release単独は、まだ始まっていないToneを取り消さない。予約抑止があるgapでrelease／continueの計算結果を
誤って再利用しないよう、窓cacheの適用条件も修正した。予定された制御更新や進行中の振幅平滑化は
固定制御として外挿せず、当該Toneの制御予測を未知とする。envelopeがゼロの区間は既知ゼロを維持する。
複合指令・追加予約・未対応制御を持つ基準は、共有単一class prototypeとの対応を`composite_default`として拒否する。

定数energyの独立した期待値、gapの両端、release tail、未知制御の発音前後、両bus、active／onset経路の
基準差とcacheを検査した。通常rendererの検査では複数On、Off／Onの順序、同batch内のOn後の取消し、
未来Update、source／body世代不一致、65 Toneの拒否を通し、observerなしの両bus音声と一致した。
既存の実観測学習との比較も複合指令へ拡張した。全Rust 1,194成功・失敗0・38ignore、31 suite、
Clippy・全target check・fmt・release buildが通過した。最終全テスト時刻は2026-09-18 21:25:54 JSTである。

通常hold／flow各4・16・64 Voiceの6条件18実行では、両bus音声、実帰結・descriptor予測・私有trace、
候補計器を除く終端観測状態、共通756件の共有観測が前版と一致した。既存18,289候補記録は追加metadataと
scope文言以外が一致し、旧未対応の複合指令6件を加えた18,295件を得た。予約を含む記録は223件だった。
1,969,528物理窓の基準・差分・支持数と共有対応拒否を検査し、renderer／instrument report間でも候補記録は一致した。
report有無の終端候補件数も一致し、容量drop・出力drop・worker拒否・worker異常はなかった。

証拠は`target/i10-known-default-20260918/`、登録は`i10-known-default.json`とfeature manifestの
`i10_known_default`である。初期compile失敗と修正前の検査失敗もartifactに保持した。
この対応はsampled batchの既知状態に限定する。最初の適格recipeが実際には拒否された場合の次recipe選択、
Off単独batch、予約だけでidleの身体、同一hopで後から到着する同sourceの別batch、未予約の未来機会は未対応である。
制御更新自体の解析予測、全将来方策、非energy帰結、身体prototype転用の適格性、共有／共同推論の残接続、
全資源・長時間・実機受入も未完了である。以前の40 ms／8.5333 ms負荷条件の失敗は解消扱いにしない。
I10未完了、I9保留・I11作用無効、既定のat-action評価を維持する。

### hop内の全指令適用後のdefault取得と実受理recipeの選択（2026-09-18）

前節のbatch単位の送出を、hop内の全batchとUpdate予約を適用した後の送出へ変更した。
Bankは既存poolのpacketを一source一件だけ保留し、事前確保した64件のVecDequeを再利用する。
候補のTone状態取得は最終送出直前の一度とし、後続batchのOff／Update・routing・世代を取り落とさない。
既存active previewの後に適格なOnが受理された場合は、そのpacketを同じpool内で焦点recipeの予測へ置き換える。
先頭の宣言recipeがduplicateやinvalidで拒否されても、次に実際に受理された適格recipeを選ぶ。
同hopの宣言recipe数は合算し、最初の適格宣言・実際の先頭受理指令・焦点Toneを区別して保持する。
policyと時計の標本は最初のsampled batchで固定したままとし、新たな将来方策や乱数評価は加えない。

従来は指令なしbatchだけを扱っていたactive previewも、OffやUpdateを含むbatchへ拡張した。
実指令適用後にも活動が残る身体の既知軌道をContinue基準として比較し、先頭の実Releaseは別記録に保持する。
活動が残らない身体にContinueを作らず、世代が一致しない身体も拒否する。
受理済み指令を含むactive基準を単一classの共有prototypeへ読み替えないよう、当該対応を拒否する条件も追加した。
実指令・実音・通常の観測学習には変更を加えていない。

通常rendererの検査で、先頭duplicate／invalidの後の受理、後続batchのOff／Update、同hopのrecipe件数、
一source一送出を確認した。receiptを除いた対照との両bus音声、実帰結、予測学習、私有trace学習が一致した。
実body Captureを使う検査では、Off単独、後続Off／Update／On、元のTone routing、正のrelease tail、
構造的な身体世代変更の拒否を確認し、observerなしの音声と一致した。
全Rust 1,196成功・失敗0・38ignore、31 suite、Clippy・全target check・fmt・release buildが通過した。
最終全テスト時刻は2026-09-18 21:41:22 JSTである。

hold／flow各4・16・64 Voiceの6条件18実行では、両bus音声、実帰結・descriptor予測・私有trace、
候補計器を除く終端観測状態、共通756件の共有観測が前版と一致した。
既存18,295候補記録はscope文言以外が一致し、指令後の活動身体167件を追加した18,462件を得た。
追加件数はflowの4／16／64 Voiceで8／34／125件である。2,002,920物理窓の基準・差分・支持数と
共有対応拒否を照合し、renderer／instrumentの候補記録、report有無の終端件数も一致した。
容量drop・出力drop・worker拒否・worker異常はなかった。

同じ短期実行の64 Voice全hop p99はholdでreportあり10.6048 ms／なし10.3377 ms、
flowで18.7502 ms／15.0939 msだった。私有100 ms完全窓のp99区間はholdで(40,50] ms、
flowで(50,75] msだった。40 ms／8.5333 ms条件は引き続き未達である。
これらは各条件一実行のwall時間であり、全hopにはoffline同期の待ちを含む。CPU時間や実機の合格としない。

証拠は`target/i10-hop-default-20260918/`、登録は`i10-hop-default.json`とfeature manifestの
`i10_hop_default`である。身体世代を変えないbrightness変更を使った初期検査の失敗も保持し、
修正版ではkind変更と実tokenの世代差を先に確認した。
予約だけでidleの身体、未予約の未来機会、制御更新自体の解析予測、full future policy default、
非energy帰結、身体転用の適格性、共有／共同推論の残接続、全資源・長時間・実機受入は未完了である。
I10未完了、I9保留・I11作用無効、既定のat-action評価を維持する。

### 候補とdefaultの自声／環境energy比率診断（2026-09-18）

実指令だけに接続されていた重なり・可聴性proxyの数値核を、実身体候補の通常reportにも接続した。
候補ごとの既存4秒・16窓のhabitat energyを保持し、窓中心間の線形補間と支持端内の値保持を行う。
このモデルは`frozen_body_window_interpolation_v1`であり、動作条件付きの残差学習モデルでも、
実envelopeの支持を直接評価する最終footprintでもない。既存の位相積を含むenergy近似の限界を引き継ぎ、
直接envelope、8／32点・密な積分、実音との順位監査を完了したとは扱わない。

環境側はVoiceの局所的な自声除外予測を使い、source ID・source世代・body世代・発行sampleの一致を要求する。
実指令用のcontext取得を維持し、指令のない活動身体にも既存20 Hzの標本時だけ取得を追加した。
64要素のstack maskと既存の予約済みcontext cacheを使い、forecastは既存poolのpacketへ値として凍結する。
poolの取得・返却時に消去し、世代不一致・発行時刻不一致・body世代不明は環境支持なしとして残す。
候補計算は既存background worker内で行うが、環境予測の取得にはhop内の追加計算費用がある。
共有busの混合予測やpresentation側の架空の環境予測へ置き換えない。

候補とdefaultは同じ4秒の物理窓と発行時の環境予測を使い、支持された絶対区間上の16等重み中点を共有する。
環境の途中の未知区間は、16点の外側にあっても支持不足として拒否する。自声の全区間支持も検査する。
両比率に同じepsilon=1e-12と正の自声積分を使い、両側が支持された座標だけ差分を返す。
全窓と交差区間の区分線形積分から除外モデル質量を記録するが、実音やモデル外の尾部の質量を主張しない。
予測値ゼロは既知の無音とは扱わず、比率・差分を未定義のまま保持する。
窓座標・モデル値・中点の値・支持・除外量をreportに残し、モデルの違いを明記した。
通常の指令予測、実観測学習、私有trace、生成方策、未実行候補を学習しない境界は維持する。

定数の閉形式、候補とdefaultの支持の非対称、隠れた環境の支持欠落、外側の切捨て、巨大な絶対時計、
未知とゼロ、identity／世代／発行時刻、標本時計の境界をRustで検査した。
全Rust 1,198成功・失敗0・38ignore、31 suite、Clippy・全target check・fmt・release buildが通過した。
最終全テスト時刻は2026-09-18 22:03:21 JSTである。

hold／flow各4・16・64 Voiceの6条件18実行を前節の凍結binaryと比較した。
出力WAV、実帰結・descriptor予測・私有trace、候補計器を除く終端観測状態、共通756件の共有観測は一致した。
既存18,462候補記録と2,002,920物理窓の数値は不変で、renderer／instrument report間の追加診断も一致した。
report有無の終端件数も一致し、容量drop・出力drop・worker拒否・worker異常はなかった。
250,365候補の比率診断を独立した区分線形segmentの端点積分と中点評価で照合した。
支持あり250,065、環境未取得108、ゼロ予測192であり、候補／defaultの非ゼロ差分は229,498件だった。
環境予測が得られた記録は18,459件で、活動身体18,239件の全件とonset223件中220件を含む。
外側の切捨てがある候補は170,892件だった。これはモデル診断の支持であり、知覚的な可聴性の成立ではない。

独立検査の初回は、約38.25のモデル積分から約0.002の除外質量を引く箇所で失敗した。
差は大きな被減算値の1 ULP（約7.1e-15）だった。両側32区分の積分和の誤差を考慮し、
除外質量だけには被減算値の64 ULP、除外率には1の64 ULPの絶対誤差枠を使った。
その他の値は相対3e-12／絶対1e-15を維持し、source・binary・取得済みreportは変更しなかった。
初回失敗のlogと根拠をartifactに保持した。

負荷は悪化した。同じ短期実行の64 Voice全hop p99はholdのreportあり16.7998 ms／なし16.1521 ms、
flowのreportあり24.7224 ms／なし20.6637 msだった。私有100 ms完全窓のp99区間は
holdで(40,50] ms、flowで(50,75] msであり、40 ms／8.5333 ms条件は引き続き未達である。
reportなしの個体処理区間は、全個体数が揃ったhopの平均でhold約133.5→1,022.6 us、
flow約602.0→1,544.8 usへ増加した。追加した同期環境予測の取得費用と整合するが、
一実行のwall時間比較であり、CPUプロファイルによる原因確定や正式な全負荷・長時間・実機受入ではない。
次の優先作業はこの取得費用の削減であり、診断値を保ったまま比較する。

私有release traceは個々のToneのrenderer終端を教師にしている。複数Toneの終端をVoice全体の単一時刻へ
集約する対応は未登録であり、候補release/gapへ最大終端や平均を代入して解決したとはしない。
この接続、直接envelopeに基づく比率、予約だけでidleの身体、未予約機会、将来制御とfull default、
身体転用、共有／共同推論の残接続、全資源受入は未完了である。
証拠は`target/i10-candidate-energy-ratios-20260918/`、登録は`i10-candidate-energy-ratios.json`と
feature manifestの`i10_candidate_energy_ratios`である。I10未完了、I9保留・I11作用無効、at-action評価を維持する。

### 環境energy予測の計算共用と工程の再点検（2026-09-18）

前節で増えた環境予測取得の費用に対し、二つの数値計算を変更した。
履歴予測の三帯域はfeatureごとの走査を一度にまとめ、各帯域内の加算順序と全201 horizonを維持する。
周期予測では、recent onsetが1未満、または最小lagもまだ成立しない時刻に限り、全horizonが同じ
履歴平均の順序付き和になることを使い、一度の計算を再利用する。それ以外は従来どおり各horizonを計算する。
この共通核を共有観測時の発行、実際の局所予測発行、readonlyの局所previewの三経路に使う。
入力、予測対象、学習、標本時計、容量、支持、モデルの帰結を省略する変更ではない。

950 stepの学習状態とring wrapを使う既存の独立batch ridge検査へ、旧帯域計算とのbit一致を追加した。
三つの時間解像度で最小lagの成立直前／直後、全lag成立時、長い時計、recent onsetの1直前／一致／超過を
検査し、201 horizonすべてを旧計算とbit単位で照合した。
全Rust 1,199成功・失敗0・38ignore、31 suite、Clippy・全target check・fmt・release buildが通過した。
最終全テスト時刻は2026-09-18 22:19:21 JSTである。

前節の凍結binaryを対照とするhold／flow各4・16・64 Voiceの6条件18実行も通過した。
出力WAV、実観測学習、終端状態、既存候補と追加比率を含む18,462記録・250,365候補は一致した。
共通756件の共有観測、report有無の候補終端件数も一致し、容量drop・出力drop・worker拒否・worker異常はない。
新しい源コードは`src/core/history_prediction.rs`と`src/core/temporal_expectation.rs`の二ファイルだけである。

64 Voice・reportなしの全個体数が揃ったhopについて、個体処理区間のwall平均は
holdで1,022.6→830.3 us、flowで1,544.8→1,360.8 usとなった。減少率はそれぞれ18.8%、11.9%である。
全hop p99はholdのreportあり16.6881 ms／なし15.7671 ms、flowのreportあり24.2240 ms／なし20.0990 msだった。
私有100 ms完全窓のp99区間はholdのreportあり(50,75] ms／なし(40,50] ms、flowは両方(40,50] msであり、
40 ms／8.5333 msの性能条件は引き続き未達である。個別worker窓には悪化もあり、全面的な負荷改善とはしない。
各条件一実行のwall時間であり、統計的なCPU速度向上や全8 prototype・600秒・実機の受入を示す結果ではない。

作業中、ユーザーから「工数がかかるのは計算負荷と機能を同時に満たそうとしているからか」と問われた。
同時最適化の往復に加え、後続記録でI12の共同推論やR2/A3の受入までI10へ積み増し、
途中の近似診断や細かな変更の検証を増やした進め方にも問題があると判断した。
ユーザーの質問を、要求の削除やI10完了の承認とは解釈しない。
I10の元の資源行は全費用を測定してR2へ渡すものであり、R2の全処理合格やA3実機受入とは区別されている。
今回の性能修正はここで区切り、次は機能上の残項目へ戻る。各機能追加の途中で最終性能合格まで延長し続けない。
未解決の範囲質問と全要求は保持し、機能完成・性能測定・性能合格を別々に報告する。
この見直しに伴い、初期案の追加核単独ベンチと交互反復は実施せず、通常の実行比較の範囲だけを報告した。

証拠は`target/i10-energy-preview-cost-20260918/summary.json`と同ディレクトリの凍結版・検査記録である。
feature manifestの`i10_energy_preview_cost`からsummary hashと二つのsource hashを参照する。
別の段階別登録JSONは追加せず、本台帳に結果を集約した。
直接envelopeに基づく比率、私有release/gap、実身体/defaultの未対応、残る帰結接続・身体転用、
共有／共同推論と全資源受入は引き続き未完了である。I10未完了、I9保留、I11作用無効を維持する。

### 候補の実envelope支持から比率を直接評価（2026-09-18）

4秒窓中心の線形補間を、候補の残存Tone・追加recipe・予約release・gapから作る身体支持へ置き換えた。
候補とdefaultは同じ凍結済みの自声除外環境予測と絶対時計を使い、それぞれの身体支持を共通の最大4秒範囲と交差させる。
交差区間の16等重み中点で、包含する整数sampleのenvelope・controlを直接評価する。
sineが支持される場合は既存のcarrier区間平均を使い、それ以外は明示した加算近似へ戻す。
1 sample未満の区間分割は包含sampleのcarrier評価を使う。reportの比率contextはversion2とし、
モデル名・時計対応・候補別の身体支持／交差区間／自声値／連続支持／差分を残す。
既存の物理energy窓、実指令用の比率、実音学習は変更していない。

予定releaseが遅れて適用される場合、適用前の身体支持を消さない。gapは予約onsetの半開区間で判定する。
離れた短音の間を補間せず、16点が全てゼロでも身体支持がある場合は既知無音と扱わない。
標本点間に隠れた未知controlやSeqGateの不確実な境界も連続支持の不足として拒否する。
範囲外へ切り捨てた時間は明示し、その物理energy質量は未知とする。旧補間モデルの積分を流用しない。

遅延した1／8／17／256 sampleの短音、release適用前後、gap端、routing、隠れた未知と
標本漏れ、位相一致／相殺を検査した。512 sampleの二次releaseについて、8／16／32点を
閉形式のenergyと全sample参照へ照合した。16点の積分は21.3125、参照は21.3958740234375
（約0.39%差）、overlap差は約0.0005264だった。この条件だけの数値検査であり、任意の実音や候補順位の保証ではない。
全Rust 1,205成功・失敗0・38 ignored、31 suite。cargo test exit=0 @ 2026-09-18T22:45:53+09:00。
Clippy、全target check、fmt、release buildも通過した。

通常実行はhold／flow各4 Voiceの2条件・render／report／reportなし計6実行に限定した。
音声、実観測学習、比率以外の既存878候補記録・95,856物理窓、共通252共有観測は前版と一致した。
renderer／instrument間の新比率記録とreport有無の終端件数も一致し、worker異常や容量dropはなかった。
11,982比率候補について、記録された支持・中点座標・比率・default差を独立再計算した。
支持あり11,982、既知無音0、標本ゼロだが無音未証明0、
環境未取得0。記録値からの比率再計算であり、実身体状態からの予測全体の独立再構築とは区別する。

全個体数の反復負荷試験は行わず、性能合格を今回の機能終了条件へ追加しなかった。
証拠は`target/i10-direct-body-footprint-20260918/summary.json`、feature manifestの`i10_direct_body_footprint`から参照する。
実音全体の8／32点・密な積分・順位監査、私有release／gapの帰結fit、idle予約・将来方策default、
身体転用と残る帰結接続は未完了。I12／R2／A3の接続・性能受入も別途残る。
I10未完了、I9保留、I11作用無効、at-action評価を維持する。次は私有release／gapと実身体の終端対応を確認する。

### 私有release headと候補のTone別終端を接続（2026-09-18）

通常candidate packetの発行時に、現在のhabitat reference／anchorと学習済みrelease headを読み取り専用で凍結する。
新しいpending指令やtraceを作らず、実行・学習のissue／observeを呼ばない。元のintrinsic periodを使い、
候補時計用に丸めたsample数から復元しない。新しい照会originのcommand IDはnullとし、既存onset originの数値IDは維持する。
source／世代・head・発行時刻・busをworkerで検査し、既存の身体世代境界とともに通常reportへ渡す。

候補とdefaultのrenderer終端を、同じTone IDごとに対応させる。複数Toneを最大時刻や平均時刻へ潰さない。
予定releaseの適用時点を含め、rendererのis_doneが初めて成立する時刻を求める。
release時間ゼロは「以後のenergy支持なし」と「現在sampleに終端あり」を区別する。
既に過ぎた終端を現在へ動かさず、gapで除外した予約onsetや省略したrecipeは候補イベントなしとする。
発行から4秒以内で双方の終端と私有支持がある場合だけ、既存の両モデルのfitとdefault差を返す。
自然なenvelope終端へのrelease headの転用はラベル付きの条件付き診断であり、妥当性・Voice全体の集約・生成圧力は未確定とする。

周期／非周期referenceで実際のrelease指令発行時の凍結値との一致、照会を繰り返してもpending・学習が増えないこと、
世代・head・時刻違い、古い／未来のinventory、退役後も古い凍結値が不変であることを検査した。
複数Toneの異なるtail、gap端、routing、範囲外、即時／遅延適用も検査した。
既存renderer検査では3 backend×4 routingの実終端と新しい予測終端が一致した。
全Rust 1,209成功・失敗0・38 ignored、31 suite。cargo test exit=0 @ 2026-09-18T23:04:04+09:00。
Clippy・全target check・fmt・release build通過。初回の検査コードの借用エラーと修正後のlogを保持した。

通常検査はhold／flow／metric／立ち上がりと減衰を明確にしたmetricの各4 Voice。
各render／report／reportなしの12実行と、追加二素材の凍結旧binaryによるrenderを合わせて14実行した。
音声、実観測学習、全2,330候補記録の既存energy／比率／onset trace／共有項、
300,688物理窓、共通652共有観測は前版と一致した。
renderer／instrumentの新診断とreport有無の件数も一致し、容量drop・worker異常はなかった。
Tone別終端pairは78,880件、支持された条件付きfitは6,836件、
非ゼロ差は1,466件。既に発行済みのOffと後の実終端の698対応、
元のintrinsic periodの2330対応も独立照合した。これは自然終端への統計的転用や可聴性の検証を代用しない。

最初のflow素材では学習済みrelease headがあっても、非周期anchorから4 intrinsic periodを超えており、
将来pairの支持は0だった。持続音metric素材もretrieval referenceと周期peakを作らず、正例取得に失敗した。
両失敗logと変更前の要求を保存し、無音やゼロfitへ置き換えて合格扱いにはしなかった。
公開Rhaiにintrinsic periodの直接指定がないため、モデル・閾値・sourceを変えず、
attack／decay／低sustainを持つmetric素材を固定して取得した。この実音では周期参照が成立し、
release 1,665件・gap 4,740件を含む6,836件の支持されたfit、1,466件の非ゼロ差を確認した。
これは開発素材の条件を選んだ経路検証であり、任意素材での成功率や校正精度を主張しない。

証拠は`target/i10-release-trace-20260918/summary.json`、feature manifestの`i10_release_candidate_trace`から参照する。
最大全個体数・反復性能試験は実施せず、R2／A3へ残す。I10全体は未完了。
次は予約onsetだけでidleの身体を候補診断へ接続する。Voice全体の帰結集約・身体転用・将来default・残る帰結接続は保持する。
I9保留、I11作用無効、at-action評価を維持する。

### idle身体の受理済み予約onsetを候補診断へ接続（2026-09-18）

通常rendererの候補入口を、活動中だけでなく受理済み予約を持つidle身体へ拡張した。
先頭の生存予約を既存delayed_onsetで一度だけ表し、Toneを追加せず現在の予約列をdefaultにする。
`default_basis`は受理した新recipe・活動身体・既存予約を区別する。既存20 Hz・64 Tone／packet制限、
source／body世代境界を維持し、同hopで新しい適格recipeを受理した場合は既存packetを一度だけ置き換える。

releaseは候補時点に活動がある場合だけ生成する。idle中のgapは半開区間内の予約だけを除く。
未消費の機会とみなしてwait／skipを作らず、発行済み予約へ追加のonset recipeを重ねない。
候補範囲外の予約defaultも保持し、文脈の範囲外を既知無音へ置き換えない。周期不明時は予約defaultだけとする。
既知Offの適用とrenderer終端を使い、発音前に取り消された予約を有効defaultから除く。
共有prototype表にはqueued_defaultを返し、既存予約を新規励起の比較へ誤対応させない。

実Captureを備えたScheduleRendererで、予約維持、取消、未来Update、新recipeへの置換、source世代違い、
body世代違い、容量超過の7条件×3 routingを検査した。予約待ちから活動中への移行、
音声・実帰結・予測学習・私有traceの対照一致を確認した。数値核では予約の二重加算防止、
gapの半開境界、未知／非routing、周期なし、予測範囲外、取消を検査した。
初回は検査側のCapture未設定で候補が出ず、次にsource世代拒否の計数期待を誤った。
実Captureを設定し、入口で拒否される正しい計数へ修正した。失敗logとClippyの初回指摘を保持した。

全Rust 1,213成功・失敗0・38 ignored、31 suite。cargo test exit=0 @ 2026-09-18T23:38:16+09:00。
Clippy・全target check・fmt・release build通過。
通常素材はflow／立ち上がりと減衰を持つmetric各4 Voice、render／report／reportなしの計6実行。
音声・実観測学習・既存1,306候補記録の全従来値・204,768物理窓・
共通364共有観測が前版と一致した。report有無の件数も一致し、容量drop・worker異常なし。
この2素材にidle予約はなく、通常binary比較は保存性だけの証拠。新機能の正例は上記renderer検査による。
全個体数・反復性能試験は実施しない。

証拠は`target/i10-idle-queued-20260918/summary.json`、feature manifestの`i10_idle_queued_candidates`から参照する。
I10全体は未完了。未予約機会を含むdefault、実身体に対応した非energy帰結、身体転用と残る通常接続を維持する。
次は残る帰結の入力から通常診断までを確認し、既存数値核と実身体の対応を埋める。I12／R2／A3は別工程として保持する。
I9保留、I11作用無効、at-action評価を維持する。

### 実Voiceの判断時計と共有表の発行時計を対応（2026-09-18）

非energy帰結の通常消費を調べ、descriptor転用の妥当性とは別の時計不一致を特定した。
共有表の発行後に実Voiceがcontinueをdefaultとすると、共有時計では正のoffsetになる。
profileのcontinueはoffset 0だけに登録されており、有効なgroup対応があってもunknown_cellになっていた。
保存版の110 Hz素材で60件、直前の現行binaryで再取得して64件を確認した。

lookup version 2は最寄りwhole cellを一度選び、その共有時計における音響classを使う。
onset-nowが正のcellに当たればdelayed-onset、delayed-onsetがcell 0に当たればonset-nowを参照する。
continue／skipの正offsetとwaitのcell 0については、continue(0)・skip(0)・選択waitが
同一の保存特徴軌道IDを持つ場合だけ対応させる。実Voice側のclass・時刻・機会の消費は変更しない。
対応先が未知なら元classや隣の既知cellへ逃がさず、支持なしを保つ。世代・bus・model・鮮度の門も維持する。
型変換した数値を作らず、元cellの分布・支持・時刻・groupを一組で返し、source classを記録する。

単体検査では二時計のずれ、逆方向のonset対応、同一軌道でない場合、選択cell欠落、旧lookup版の拒否、
元の機会管理を検査した。通常Table::publicationでも、異なる評価遅延設定とbus 1の対応を検査し、
選択結果を元の表cell全体のserializationと一致させた。既存bus 0・whole-cell・鮮度・片側支持の検査も通過した。
全Rust 1,215成功・失敗0・38 ignored、31 suite。cargo test exit=0 @ 2026-09-18T23:55:51+09:00。
Clippy・全target check・fmt・release build通過。

通常素材は既存の110 Hz持続音1条件。凍結旧binaryのrender、修正版render、instrumentのreportあり／なし、計4実行。
音声、実観測学習、全37候補記録の身体energy・比率・私有trace、共通52共有観測は不変。
共有consumerはpaired 64→128、
unknown_cell 64→0。
回復した64組を、元profileの軌道ID、絶対時刻・最寄りcell・class・支持・差分式へ独立照合した。
以前から支持されたpairは全項目一致し、report有無の件数も一致。非ゼロordinal差は0件で、
支持されたゼロを非ゼロ効果や校正成功へ読み替えない。

証拠は`target/i10-context-clock-20260918/summary.json`、feature manifestの`i10_consumer_context_clock`。
今回の修正は表の時計対応であり、実身体へのprototype転用失敗やrelation／invitation／全代替のuncertaintyを解決していない。
これらと残るdefault接続を次の対象として保持する。I10未完了、I9保留、I11作用無効。R2／A3の全負荷受入は別工程。

### prototype取得側のroutingを通常consumerで検査（2026-09-19）

前回は工数の説明のみで、実装進展はなかった。今回は実身体転用の既存失敗を読み直し、
通常loaderが取得時のsource bus／recipe routingを捨て、consumerが実Voice側のroutingだけを
検査する欠落を修正した。共有表の数値や最寄り距離を補正して適合扱いにはしない。

loaderは取得busが0/1であり、元descriptorのbusと一致すること、recipe routingが登録した
both／habitat／presentationのいずれかであることを要求する。元取得busがroutingされていたかを
profileごとに保持し、snapshotとlookup version 3のkeyへ8-bit mapを載せる。
実Voiceのdefaultが未routingなら従来のunrouted_default、実Voiceがrouting済みでも元profileの
取得busが未routingならunrouted_prototypeとして、帰結差を返さない。別busの軌道への差替えは行わない。
公開時にはprofile hash・body model・routing mapの一致も検査し、別モデルの情報を古い表へ結び付けない。

新しいloader回帰検査は旧実装で失敗した。両busと三routing、欠落／不正／不一致の7条件、
実candidate workerで実Voiceとprototypeのroutingの全8組合せ、bitの独立性、旧lookup版拒否、
評価時刻設定を跨ぐ通常Table公開、元cell全体との一致、古い公開値の不変性を検査した。
途中のKey変換削除によるcompile failureは保存し、snapshotから同じkeyを作る通常経路を保つ形で修正した。
対象13検査成功・1 ignored。全Rust 1,217成功・失敗0・38 ignored、31 suite。
cargo test exit=0 @ 2026-09-19T00:18:37+09:00。fmt・Clippy・全target check・release build通過。

前版の凍結source hashとbinary結果を照合し、既存110 Hz持続音1条件を修正版render／instrumentの
reportあり・なしで計3実行した。音声、実観測学習、37候補記録の身体energy・比率・私有trace、
共通52共有観測の値は不変。routing metadataと計器のサイズ差だけを区別し、保存profile容量の
追加64 bytesも記録した。既存paired 128組は全項目一致し、この素材で新しいroute拒否はなかった。
report有無の終端件数は一致し、drop・worker異常なし。非ゼロordinal差は引き続き0件。

保存済み転用比較72 ownerの取得metadataから新しい門の作用を独立に照合した。
実Voiceがrouting済み・prototype取得側が未routingの24件を拒否し、実Voice側未routing24件と
routing一致24件を別集計した。これは保存assignmentへの門の再適用であり、24件の通常consumerを
新規実行した証拠ではない。拒否24件すべてで、距離0.25以内の別のrouting済みprototypeは存在しない。
routing一致24件には、0.25／1／2／4秒窓で各1,872候補対中280／356／304／208のsource-energy順位反転が残る。
従ってroutingだけでは転用は成立せず、残った24件を採用扱いにしない。これはordinal順位の結果でもない。

証拠は`target/i10-prototype-routing-20260919/summary.json`、feature manifestの`i10_prototype_source_routing`。
I10未完了。次はrouting一致の比較で残る実recipe・既存Tone・control・位相とprofile応答の差を
実身体の候補帰結へ反映する経路を埋める。関係支持・invitation・全代替uncertaintyと残るdefaultの
通常接続も保持し、今回の拒否門をその代替にはしない。I9保留、I11作用無効、at-action評価を維持する。
R2全負荷合格・A3実機受入は別工程であり、今回の短い実行から完了を主張しない。

### I10の残依存を現行sourceへ照合（2026-09-19）

prototype routing修正後の227 source fileを保存hashへ照合し、最新の全Rust 1,217成功の
report/statusが保存版と同一であることを確認した。そのreportから、実音・身体記述・予測学習・
私有trace・候補defaultに対応する18検査の成功を個別に照合した。新しいRust変更や再実行はない。
これは完成監査の依存確認であり、成功件数をI10全体の完成証拠にはしない。

| 残項目 | 現行sourceで確認した事実 | 単なる配線で済まない理由 |
| --- | --- | --- |
| 共同posterior | `temporal_cognition.rs` は `joint` 全体を `cfg(test)` に限定する | 通常観測の共有contextと条件付きlocal pathの所有者、実proposal、登録した特徴score入力がない。独立した周辺分布の積で代用しない |
| 全帰結の射影 | `consequence` も `cfg(test)` に限定する | 数値核の検査はあるが、実際の同時代替とaction score差を通常workerから渡す経路がない |
| 参加意欲head | `ratings::groove_features` の54入力／109座標組立も `cfg(test)`。通常headなし | 観測由来の時刻履歴と特徴・明示head設定・候補射影が必要。Voiceの私有参加traceは別機構であり代用不可 |
| 現在の共有帰結 | `RawHeadMixture` はclosure／continuationの二列。通常projectionは「Raw base-head aggregation only」と明記 | 発行時phrase質量を保持するraw混合であり、共同score差による再重み付けではない。評定categoryのentropyも解釈uncertaintyとは異なる |
| 実身体転用 | routing一致の保存24 ownerにも候補energy順位反転が残る | route検査、距離門、代数の一致だけでは身体応答の転用を証明しない |
| 資源引渡し | 4／16／64 Voiceの既存測定は保存した各版の実装済み処理を対象にする | 未実装処理をゼロとして最終I10費用を確定できない。R2全処理合格とA3実機受入は別工程 |

再開時の誤りを防ぐため、上記を「既存数値核を呼べば終わる残配線」とは扱わない。
検査可能な証拠・source位置・hash・残依存は
`target/i10-completion-dependencies-20260919/audit.json` に保存した。
これは過去の成功の取消でも、既存要件の削除でもない。

工程表のI10境界と後続記録で追加した共同推論・全headの範囲が一致していないため、
ユーザーへ完了範囲を再確認した。質問は、工程表の身体→私有trace・候補数値→診断の境界でI10を閉じ、
後段の共同推論・head統合を別工程へ残すか、それらの通常接続まで今回のI10へ含めるか、の二案。
両案ともR2／A3は別工程であり、工程表案を選んでも残要件の完成監査は必要である。
回答前に未実装列を完了扱いにしたり、I10を完了にしたりしない。

この判断待ちは共同推論・head追加の工程帰属に関するもの。既存の実身体/defaultの要件は保留しない。
独立に進められる候補は、確定した未来control更新が一つあるだけで、更新前の支持されたenergy予測まで
全窓未知になる経路の修正。`Tone::stable_prediction_control` がpending updateの存在だけでNoneを返すため、
実時刻の支持境界を保持する必要がある。未来更新後を推測して既知扱いにはしない。
I10未完了、I9保留、I11作用無効、at-action評価を維持する。

## 確定したcontrol更新より前の予測支持を保持（2026-09-19）

I10の実身体/default経路に限定した修正。未来のTone更新が一つあるだけで全control予測をNoneにする処理を、
最初の予約更新の絶対sampleを有効期限とする処理に変えた。更新前は既存の条件付きenergy予測を返し、
更新時刻以降は未知を保持する。遅延onset候補でも予約更新の期限は移動しない。
現在進行中のamp smoothingは未支持のまま。共有prototypeへの全体転用も、部分支持を理由に許可しない。

旧実装で境界回帰の失敗を確認。実rendererでsample 448発行・2500更新のケースを追加し、
[448,2448)の250 ms窓の既知予測と、その後を含む1/2/4秒窓の未知を確認した。
両busの音声と、候補評価の有無による実予測・私有trace更新の一致も検査した。
空区間、期限ちょうど、予約の並び順、release後の既知無音を検査した。
これは既存の加算近似の支持回復であり、pending updateのあるsine波形の正確な予測ではない。

全Rust 1,219成功・失敗0、fmt/clippy/all-targets/release通過。
通常110 Hzの一素材を一回取得し、前版と音声、実学習、37候補値、共通52観測が一致した。
全Voice性能行列は再実行していない。記録: `target/i10-control-support-20260919/summary.json`、SHA256 `865d1591c08c9ef94054be4e2efff1f58d8f71e98290a67a91ebf2f5a2f9a8be`。

I10全体は未完了。共同推論・全headの工程帰属は既に提示した範囲質問の回答待ちであり、
実身体へのprototype転用の失敗も残る。I9保留、I11無効、at-action評価を維持する。

## 通常の遅延onset候補の位相を実rendererへ一致（2026-09-19）

実身体への転用監査を起点に、検証用の七class予測と通常candidate workerの不一致を発見した。
実rendererは新Toneのsource ID・onset sample・Tone IDから初期位相を決める。検証用経路は
遅延onsetごとに位相を作り直す一方、通常経路は発音時刻だけ移し、defaultの位相を使い回していた。
旧実装の通常経路で、250 Hz・sample 1239の予測+0.299490と実音-0.297254の反例を再現した。

seedから位相状態を作る処理をrenderer・予測・遅延候補で共用し、通常経路も候補onsetを使う。
元のdefaultと残存Toneを再seedしない。帯域外のゼロcarrierも維持する。
検証用経路も同じ遅延onset helperを使い、独立したrenderer分岐との比較を通常workerのテストへ追加した。
2周波数×12候補時刻×最初の64 sample×両busの3,072比較で、最大絶対誤差は
1.92907390228e-07。既存sine予測テストの許容0.002を検査前に固定した。
この値は対象sampleの誤差であり、全4秒窓や全backendの精度を示さない。

通常110 Hz flowの旧版・新版二実行で、実音・実予測学習・私有traceが一致した。
146候補記録中、遅延onsetの66候補で身体energy窓または直接energy比率が変化した。
それ以外の候補、候補metadata、私有trace帰結、共有headは一致。共有観測の共通162件も一致した。
全Rust 1,220成功・失敗0、fmt/clippy/all-targets/release通過。全Voice性能行列の再実行なし。
記録: `target/i10-candidate-phase-20260919/summary.json`、SHA256 `f5bd329e07083290204248fe2222c7277cf5a3b076a741afeb88e675a4da1e79`。

I10未完了。今回の誤りは通常の実身体モデル内の位相対応であり、六記述値からのprototype転用の
反例や非energy帰結の未接続を解消したとは扱わない。I9保留、I11無効、at-action評価を維持する。


## 実音群の相対時刻履歴を通常観測へ接続（2026-09-19）

参加意欲等の残る帰結入力に必要な、公共の音群由来の相対時刻履歴を実装した。
実Voiceの私有参加traceは別機構であり、今回の入力に流用しない。通常recurrenceが受理した
accentとその更新後の周期候補を履歴producerへ渡し、producerは更新前に保持した周期・版で
現在batchを採点してから新候補を選ぶ。同時accentの参照は全音群について一括で固定する。
非周期参照は実観測済みの直近2〜4間隔の中央値で、新しいscaleはその記録だけへ固定する。
音群内は厳密な直前accentで固定した周期・版を使用し、欠測区間や版変更を跨いだ標本を作らない。

7音群の順序付き42組に各2 familyと、独立した7音群内履歴を持つ。各履歴は32 bin、128件、
8秒の物理窓。既知hopの一様timestamp差を積分し、周期だけをwrapする。非周期の範囲外質量は別保持。
音群間の重みは両accent重みの積、音群内は小さい方。前者は不確実な両入力を乗じる入力規則として
今回のplanへ固定した。絶対sampleを整数で差し引いてから正規化し、大きいclockの桁落ちを避ける。
周期の交代は支持比1.25以上が連続1秒、観測coverage90%以上。欠測・挑戦候補交代・比較不成立で
待ち時間をリセットする。支持を失った現周期の履歴は休眠し、交代時に破棄数を記録する。
容量退避は窓coverageから失われた前史を除外し、90%未満を支持扱いしない。退役した音群の行・列は消す。
reportは支持重量×現在参照支持の上位2音群間履歴と、独立した音群内履歴を出す。
modeや14値写像を実装した報告ではなく、その入力となる記録・正規化histogramの診断である。

10個の数値・因果・境界テストを追加した。32 binの積分は独立した401×401点のtimestamp分布と比較し、
同時刻・wrap・linear overflow・大きいclock、周期差替、非周期中央値、欠測、容量・世代を検査した。
通常Frontend→Recurrenceの追加fixtureでは支持638 frame・最大27記録を確認した。
保持容量は伸びず、1記録56 byte。48 kHz／512 hopで1 bus当たり745,152 byteの所有量を診断へ公開する。
この値は新producerの所有量であり、全runtimeのメモリ使用量ではない。

実音の取得は新規5実行に限定した。flowは保存済み前版との比較、pulseと振幅変化素材は各新旧一回。
flowの支持0件、pulseの最大coverage32.75%・支持0件を保持し、閾値は変更していない。
同じ音高を保って振幅だけを変える素材では、音群内周期履歴が両bus各3件、保持1間隔で支持された。
Scenarioの時刻表はobserverに渡さず、音響観測だけで更新した。音群間／非周期の通常PCM正例は未取得。
3素材すべてで音声・実学習・既存754候補・共通806観測が一致した。今回の支持正例は入力接続の証拠であり、
参加意欲の予測精度や聴取者の区別を示すものではない。

全Rust 1,231成功・失敗0・38 ignored、fmt/clippy/all-targets/release通過。
初回の全テストでは既存のrender比較テストが、一度開けた一時WAVを後で読めず失敗した。
原因は未特定。失敗ログを保存し、同テスト単体と全テストの再実行はsource変更なしで通過した。
証拠は`target/i10-auditory-timing-20260919/summary.json`、
SHA256 `76d4f64592b15662fe137a41095576f7037c4baf0da6e5dd7a43f5aae183ac5d`。

I10未完了。次は登録済みのplateau mode・nearest-mode dispersion・42値写像をこの履歴から作る。
残る12生特徴、groove/desire head、候補射影、共同推論、実身体prototype転用は未完了のまま。
全headの工程帰属に関する既存の質問は未回答であり、この入力実装で回答を代用しない。
I9保留、I11作用無効、at-action評価を維持する。R2全負荷・A3実機受入は別工程であり、今回は実施していない。


## 音群の時刻履歴から42生特徴を通常診断へ接続（2026-09-19）

登録済みの時刻履歴から、音群間の上位2履歴と独立した音群内履歴の各14生特徴を通常reportへ接続した。
正規化f64 bin値のexact-equal plateauを一つのmodeとし、周期では境界を跨ぐplateau、非周期では
片側端点を扱う。modeは代表binの質量と安定bin IDで並べる。一様なhistogramはmode数0を既知とし、
分散を欠測とする。支持なしの履歴は全14値を欠測、存在しないmodeはその4値を欠測とする。
周期のlinear座標、非周期のcos／sinは、modeが存在する場合だけ適用外の既知ゼロとなる。
2履歴の重みを選択された支持量で正規化し、音群内履歴の重みは1とする。残り12生特徴をゼロで埋めない。

残差分散はbin中心に記録を丸めた近似ではなく、保存したtimestamp差分布の二乗距離の期待値とする。
差分PDFの区分線形区間と全modeの最近傍領域に分割し、各区間の三次多項式を積分する。
周期は最短円距離を0.5 cycle、非周期はlinear距離を4で割る。非周期の範囲外質量も分散へ含め、
周期化や1への切詰めは行わない。表示する上位2 mode以外も、mode数と最近傍分散に含める。
履歴のshapeは選択後に計算してcacheし、insert／expire／resetで無効化する。記録は56 byteのまま、
48 kHz／512 hopで1 bus当たりの所有量は745,152から749,520 byteへ増えた。全runtimeの資源受入ではない。

6検査を追加し、通常recurrenceの既存正例にも特徴値の検査を追加した。uniform、plateau、wrapped plateau、
端点、1 ulpの差、16 mode、第三modeが最短になる例、overflow、cache失効、3ブロックの順序と欠測を検査した。
連続分散は一様分布の閉形式momentと一致。独立した16条件×601×601のtimestamp点積分との最大差は
5.26167082459672e-07で、事前の検査許容0.00002を満たした。

通常実音は既存の振幅変化素材を新版で一回だけ取得した。両bus各3件の支持あり音群内周期履歴から、
計60既知座標を確認した。同じ履歴のmodeを、equal-bin連結成分を用いる独立Python実装で再計算し、
modeと14値写像の一致を確認した。通常reportの連続分散は上記Rustの独立moment／分布検査で支え、
reportの間引かれたaccentだけから全原timestampを復元したとは主張しない。
音声・実学習・318候補・共通332観測・元の履歴は不変。今回の通常正例は1間隔を保持する音群内周期履歴に限り、
音群間／非周期の通常PCM正例や、聴取者の判断・headの予測精度を示すものではない。

全Rust 1,237成功・失敗0・38 ignored、fmt/clippy/all-targets/release通過。全Voice負荷行列は実行していない。
証拠は`target/i10-timing-features-20260919/summary.json`、
SHA256 `a9df3a7b993cb611d819fee8a4b8fb2cbff24999e1478e000e09efe7d108cb00`。

I10未完了。42時刻入力の次は、8密度窓・word entropy／その二乗／surprise・grouping平均の12入力と、
groove／desireの独立head、候補射影を接続する。共同推論・実身体prototype転用・残るdefaultは未完了。
既存の全headの工程帰属に関する質問は未回答のまま。I9保留、I11無効、at-action評価とR2／A3の別工程を維持する。


## 残る12生入力と54入力の通常診断を接続（2026-09-19）

既存42時刻入力に、8密度窓、word entropy／二乗／遷移surprise、8秒grouping平均を加え、
通常recurrenceの音群ごとの`groove.raw`へ54入力を順序固定で接続した。新しいVoice私有入力は使わない。
密度のaccent数は既存台帳の重み和とし、各窓の実観測秒数で割る。時刻はepochで切り詰めた物理窓を使う。
128件容量退避や保存期間不足を既知ゼロへ変換せず、各窓の保持支持と観測率を別に報告する。

wordは音響groupingの連続accent列から取り出す。同じ長さ・順序付きsymbol・元anchorを持つ仮説を
period bin間で重複加算せず、最大admission supportへ統合して正規化する。wordのsymbol頻度とgrouping値は
音響assignment重みで時間積分し、8秒のvalid supportが90%未満なら欠測とする。groupingは登録したrefresh間で保持する。
新しいaccentで古いwordが不一致になった期間は、次の一致refreshまでword支持を与えない。

遷移表は音群世代ごとの9×9 f64累積数。元の連続二間隔が完全観測された新規対だけを、元の観測順で
score-before-updateする。同じ対の不確実な仮説は総更新重み1を分け、再解釈では同じ対を再学習しない。
対応wordがなかった既往対も、後のrefreshから遡って加算しない。直近surpriseは128対に制限し、容量損失が
8秒窓と交わればsurpriseを欠測とする。世代交代で統計・履歴を消去する。48 kHz／512 hopの所有量は
1音群195,040 byte、7枠で1 bus当たり1,365,280 byte。既存の時刻履歴や他workerの量は含めない。

6検査を追加し、既存の通常音響recurrence検査も拡張した。全8密度窓の129 accent対照、容量退避後の回復、
観測秒数による密度、assignment重み付きentropy／groupingの閉形式、90%境界、既知ゼロと欠測、
複数仮説の事前surprise、複数対の順序、同一対の再解釈、観測gap、word重複、世代交代、保持量を検査した。
通常音響fixtureでは密度707 frame・grouping平均8 frame・約26組の遷移学習を確認し、EOFによる支持の捏造がないことも検査した。

初版は全Rustテストを通過したが、通常release実行が観測workerのstack overflowで失敗した。
GDBでfrontend→producer→Trackerの深い呼出しと大きな診断一時領域の重なりを確認した。
`Recurrence::consume_acoustic`を非inlineの後処理に分け、音響解析と診断集計のstack寿命を分離した。
threadのstack上限は変更していない。失敗binary・ログ・当時の検査とsource hash、GDB記録、修正前後の
release命令を保存した。修正版で必須検査と通常実行をやり直した。

修正版の振幅変化素材は通常一実行で完走した。両busで計1,990音群記録に密度、24記録にgrouping平均を得た。
16秒窓の64件は累積accent重みと物理観測時間から独立再計算して一致した。
wordの累積学習は最大2組だが、8秒のvalid supportは最大0.17567567567567569であり、
entropy／surpriseの支持あり実音正例は0件のまま。数値検査の正例を実音での支持達成へ読み替えない。
音声・実学習・318候補・全332共通観測・既存時刻特徴は前版と一致した。

全Rust 1,243成功・失敗0・38 ignored、fmt/clippy/all-targets/release通過。
cargo test exit=0 @ 2026-09-19T02:14:57+09:00。
証拠は`target/i10-groove-inputs-20260919/summary.json`、SHA256 `820aa5c378e73fb1067199d5e3b6144649bed8c9bc37eb01a6660fefba47b377`。

I10全体は未完了。54入力の次は独立groove／desire headの明示設定と通常消費、候補射影が残る。
共同推論、実身体prototype転用、残るdefaultと全headの工程帰属に関する未回答も維持する。
I9保留、I11無効、at-action評価とR2全負荷／A3実機受入の別工程を維持する。


## 独立groove／参加意欲headの通常診断を接続（2026-09-19）

明示的な`temporal_groove`設定を追加した。54個の平均・偏差は共有し、grooveと参加意欲はそれぞれ
109係数、順序付き四閾値、正で正規化された五カテゴリprior、正の温度を持つ。暗黙のモデルはなく、
`temporal_period`を必要とする。固定長、未知キー、有限性・順序・正規化・設定依存を検査する。
通常recurrenceの音響生入力だけを使い、Voiceの私有traceや共同事後分布は今回のheadへ混ぜない。

音群ごとに54入力を値と欠測maskの109座標へ変換し、独立したordinal分布を計算する。
全入力欠測なら切片による支持を作らない。音群重みは保持音群の音響assignment積分を正規化し、
未支持音群を除いて重みを付け直さない。分布を混合してからheadの温度を適用し、
8秒の物理観測率とhead支持の積で、温度を掛けないpriorへ戻す。観測率は元の既知sample区間の和集合であり、
音群の適格性から独立する。epoch・gap・EOFを扱い、未来の配送時計で観測率を増やさない。
標準化、予測子、温度のoverflowは別々に報告し、一方のheadの失敗で他方を欠測にしない。

reportは`acoustic_base_heads`の範囲を明示する。UIは評定、物理観測率、head支持、prior寄与、数値エラーを表示する。
UI経路のcompileは通過したが実画面の目視は未実施。48 kHz／512 hopで設定と有界coverage履歴の所有量は
1 bus当たり20,872 byte。外側snapshotや他worker、全runtimeの資源受入は含めない。

五検査を追加した。独立した確率空間での混合参照、部分欠測・全欠測、物理窓の重複排除、未知質量、
独立headと数値エラー、設定往復、EOFを検査した。通常音響fixtureの720同一入力frameで、707 frameに支持を得て、
327 frameで両headの評定が異なった。releaseの診断一時領域は別関数とし、既存の標準stack上限を維持する。

通常実音は無効／有効の二実行。有効設定は既知の非ゼロ係数を持つ未fit検査モデルとし、精度主張には使わない。
両bus・二headで計656件の支持あり評定がpriorと異なった。独立Pythonで54→109→線形予測子→ordinal→
音群混合→温度→priorを再計算し、最大誤差は予測子1.7763568394002505e-15、最終分布3.3306690738754696e-16。
物理観測率の和集合はRustのgap／epoch検査で支え、間引かれたreportから復元したとは扱わない。
音声・実学習・318候補・既存332観測は前版と一致。entropy／surpriseの実音支持不足も保持した。

全Rust 1,248成功・失敗0・38 ignored、fmt/clippy/all-targets/release通過。
cargo test exit=0 @ 2026-09-19T02:37:46+09:00。
証拠は`target/i10-groove-heads-20260919/summary.json`、SHA256 `2a81484576e29a3c99bdf742b7650673bc53aa2cd6ec7ed1da98f1126b404e27`。

I10全体は未完了。候補射影、共同事後分布、実身体prototype転用、残るdefault対応と工程帰属の未回答を維持する。
今回のhead接続は人間への適合や未来帰結の転用精度を証明しない。I9保留、I11無効、at-action評価を維持し、
R2全負荷・A3実機受入は別工程とする。


## 候補の8密度窓を通常共有候補表へ接続（2026-09-19）

通常grooveの8密度窓を、既存の共有候補表の入力診断へ接続した。窓は0.125、0.25、0.5、1、2、4、8、16秒。
終端は候補の評価時刻、始端はepochで切り詰める。発行時の音群世代・物理clock・配送clockが一致する場合だけ、
元のGroupの観測区間とEstimatorのaccent台帳を読む。退役・古いclock・未設定headは接続しない。
未来評価でも発行後の観測やword学習を取り込まず、同じ評価時刻の観測prefixは候補class間で再利用する。

候補accentの判定は既存continuation用の四hop走査を共用し、追加の未来全走査を作らない。
元の既知sample数・重みと、支持された仮の中心区間sample数・重みを別に保持する。支持が90%未満、または
accent容量退避が窓と交差する成分は欠測を維持する。未来の支持なしを無音へ変換しない。
中心が発行時刻で終わっても、判定に未来の右隣hopを使うaccentはprojected扱いとなる。
当該accentやdensityを通常観測・私有学習へ加算しない。候補headの観測率や音群混合重みはこの窓から再定義しない。
今回の追加は入力診断であり、groove／参加意欲の候補headはまだ接続していない。

三検査を追加し、既存の通常recurrence検査を拡張した。8窓×複数offsetについて独立した重み付き個数と
区間長の計算を照合し、発行時一致・欠測・128件容量・epoch・世代・観測不変を検査した。
実際の共用accent走査から正の候補accent一件を受け、plateauと終端の未来支持不足を検査した。
通常音響fixtureの5,736窓は、発行時刻で通常の8密度入力と完全一致した。

releaseの通常実音取得は新規一回。両bus合計490件の公開候補cell、1,696個の支持ありprojected値を確認した。
この490件はreportへ公開されたcellであり、内部の全cell数の主張ではない。
profileの固定binaryから、音響prefixを必要としない未来部分1,400窓の四hop支持とaccent重みを独立再計算して一致した。
対象素材の該当未来accent重みは全て0であり、正の未来accentの通常PCM正例とは扱わない。正例は上記Rust fixtureに限る。
発行前の実音prefixや右隣との境界はRust検査で支え、間引かれたreportから原観測を復元したとは扱わない。
検証器の初版は保存候補の発行clockと後のreport clockを混同して比較に失敗した。両clockが一致する場合に限定して修正し、
通常発行時一致の主証拠は5,736窓の直接検査とした。実装sourceの追加修正や実音の再取得は行っていない。

音声、実学習、318候補の既存帰結、332観測と通常groove／参加意欲headは前版と一致した。
候補表の保持量は1 bus当たり5,363,448から6,213,048 byteへ増加した（差849,600 byte）。
これは追加の8窓診断配列とその保持枠の費用であり、全runtimeの負荷受入ではない。

全Rust 1,251成功・失敗0・38 ignored、fmt/clippy/all-targets/release通過。
cargo test exit=0 @ 2026-09-19T17:38:08+09:00。
証拠は`target/i10-groove-density-projection-20260919/summary.json`、SHA256 `8ab53f33866142bb33267b2fde04ff089ee20a4468561f6f5359d95eb1bce817`。

I10未完了。候補入力の残り42時刻値と4 word／grouping値、独立した候補headの発行時混合、共同事後分布、
実身体prototype転用、残るdefault対応が残る。工程帰属の未回答も維持する。
I9保留、I11無効、at-action評価は変更せず、R2全負荷とA3実機受入は別工程とする。


## 候補の42時刻特徴を通常共有候補表へ接続（2026-09-19）

候補音群のoutgoing 12本と独立した音群内1本を、有界scratchへコピーして42特徴へ接続した。
Stream全体を複製せず、13履歴と未来支持区間の領域を候補間・構築間で再利用する。各履歴は128記録・8秒、
候補終端は最大4秒。元の観測記録、周期選択、非周期中央値、word学習、他音群の状態は更新しない。
既存の候補accent走査を密度・arrivalと共用し、新たな未来全走査を追加しない。

音群間では、発行時の実際の最新accent、支持された周期と版、または観測済み間隔の中央値を固定する。
未来の外部accentは作らない。音群内の最初の比較は実際の直前accentで凍結した参照を使い、
以後の仮のaccent同士は発行時の同じ周期を使う。欠測・支持のないcenter・区間gapは音群内の連結を切る。
候補accentで新しい周期や参照を獲得しない。実Accent型を偽の観測として更新せず、timestamp区間から
共通のRecord数式へ渡す。epoch・世代・発行clock・4秒範囲の境界も検査する。

評価時刻で終わる8秒窓へ元記録を切り詰め、仮の記録を追加する。容量退避の損失は支持から除外する。
支持区間の実観測部分と仮の部分、両者の記録数を別に保持し、90%基準を維持する。
参照支持は発行時の値のまま、音群間上位2履歴と音群内履歴を選ぶ。exact plateau、mode数、連続timestamp差の
最近傍分散、14値写像は通常と共用する。高価なshape計算は選択後の3履歴だけに行う。
仮の支持または記録に依存する座標はprojected、それ以外の既知座標はobserved、未支持は欠測として報告する。
これは候補入力の支持であり、headの発行時観測率を置き換える処理ではない。

三つの数値検査を追加した。周期・中央値基準・音群内について、追加記録と元記録を独立した
三角形timestamp差分布のCDFで再集計し、bin質量・overflow・記録数・重みを照合した。
発行時一致、欠測による連結切断、支持のない周期の維持、容量退避、scratch再利用、古いclock・世代・範囲を検査した。
通常Frontendの720 frameでは、発行時の30,114座標が既存の時刻特徴と完全一致した。
初回の合成fixtureは遅延評価が必要なtimestamp構築を先に評価してunderflowしたため、fixtureを修正した。

releaseの通常実音一回では、公開490候補の42時刻値は全て未支持だった。対応が成立した候補の時刻履歴に支持がなく、
その欠測を維持した。実音の支持あり正例とは扱わない。音声・実学習・318候補の既存値・332観測と
通常head、前版の候補密度は一致した。reportの公開cellは内部全cellの一部であり、全内部cellの支持数も主張しない。

この未支持ケースだけを接続完了の根拠にせず、通常Frontend→Gesture／Phrase→Tableを通る数値fixtureを追加した。
対応だけを検査用に固定し、二つの仮のaccentを持つprofileを使った。4候補に支持ありのprojected入力が届き、
仮の間隔4件と16座標の変化を確認した。元の音響観測とphrase状態は完全一致した。
固定対応はdescriptor距離からの実身体転用の検証を代用しない。追加はtestだけで、再buildしたrelease binaryのhashは
実音取得時と同一だった。追加後に全必須検査をやり直した。追加前の検査・source hashも保存した。

通常実音での候補表とscratch保持量は、前版6,213,048に対して最大9,945,968 byte／bus。
scratchのheap所有分は103,984 byte、Recordは56 byteのまま。残りの増分は各cellの時刻特徴・履歴診断等。
これは保持量の記録であり、全worker・64 Voiceの時間／メモリ受入ではない。

全Rust 1,254成功・失敗0・38 ignored、fmt/clippy/all-targets/release通過。
cargo test exit=0 @ 2026-09-19T22:00:22+09:00。
証拠は`target/i10-candidate-timing-features-20260919/summary.json`、SHA256 `eef0dbef2c824131dceb2a17313db2fcd41955c14bcad58b10a88b8d1cde823c`。

提示した7大工程のうち、候補42時刻特徴の接続・数値検査を閉じる。残り6大工程は、4 word／grouping入力、
groove／参加意欲の候補評定、共同推論・不確実性・関係支持、実身体への転用、実default対応、統合検査。
共同推論の工程帰属に関する未回答は維持する。I10全体は未完了。I9保留、I11無効、at-action評価は変更しない。
R2全負荷・A3実機受入は別工程として維持する。


## 候補のword／grouping 4特徴を通常共有候補表へ接続（2026-09-19）

候補評価時刻で終わるepoch-clipped 8秒窓へ、元の観測済みwordとgroupingを再集計する経路を接続した。
通常と同じ重み付きword entropy、その二乗、元のscored pairの平均surprise、grouping支持を使う。
未来のword・groupingを仮造せず、未来部分の未支持を含めた窓の90%条件を維持する。
surpriseは窓内に残る元のpairだけを集計し、容量損失が窓に残る場合やpairがない場合は欠測とする。
学習count、処理済みpair、観測frameは更新しない。既知の値はobserved、未支持はunsupportedと記録する。
これは実観測prefixからの入力であり、headの発行時観測率や音群混合重みを再定義しない。
同じ音群・発行時刻・評価時刻のcontextは候補class間で共用し、追加の未来accent全走査は作らない。

独立した二frameの重み付き数値例で4値を照合した。ちょうど90%と直後の欠測、古いpairの窓外化、
容量損失の回復、既知0、pairなし、epoch開始、過去／4秒超の拒否、resetと原状態不変を検査した。
通常Frontend 720 frameでは発行時の2,868座標が通常値と一致した。通常Gesture／Phraseを経た候補表にも、
検査用に対応だけを固定した3候補で支持ありgroupingが届いた。固定対応は実descriptor転用の検証ではない。

release通常実音は新規一回。両bus合計490件の公開candidate cellでは追加4特徴は全て未支持だった。
この素材を支持あり正例とは扱わず、正例は上記数値fixtureと区別する。公開cellは内部全cellの一部である。
音声・実学習・318候補の既存値・332観測、既存の候補密度／時刻特徴は前版と一致した。
候補表とscratchの保持量は最大9,945,968から10,305,968 byte／busへ増加した（差360,000 byte）。
全負荷性能や実身体転用の受入を示す値ではない。

全Rust 1,256成功・失敗0・38 ignored、fmt/clippy/all-targets/release通過。
cargo test exit=0 @ 2026-09-19T22:14:07+09:00。
証拠は`target/i10-candidate-word-features-20260919/summary.json`、SHA256 `1522ad6eb51ede1720e0957852e01bd3630e39fbfc3d961484c0a857576165fb`。

54個の候補raw入力（42時刻・8密度・4 word／grouping）の通常接続・数値検査が揃った。
提示した残り工程は5大工程：候補groove／参加意欲評定、共同推論・不確実性・関係支持、実身体転用、
実default対応、統合検査。共同推論の工程帰属に関する未回答は維持し、範囲を削除しない。
I10全体は未完了。I9保留、I11無効、at-action評価は変更しない。R2全負荷・A3実機受入は別工程。


## 候補groove／参加意欲の基礎評定を通常共有候補表へ接続（2026-09-19）

54個の候補raw入力を通常と同じ109座標写像・独立した二つのordinal headへ接続した。
発行時の8秒窓、物理観測率、音群混合重み、headの係数・閾値・温度・priorを固定する。
対応先の1音群だけを候補値へ置き換え、他音群の分布と未支持質量を保持する。混合後に温度を適用し、
発行時観測率×支持質量でpriorへ退避する。未来窓の未観測率で発行時観測率を置き換えない。
observed／projected／unsupportedの入力数と、選択音群のpredictor・分布・error、全体評定を別に記録する。
共同推論前の基礎診断として`conditional_acoustic_base_heads`と明記し、生成作用や転用校正済みの値とは扱わない。

通常用と候補用の局所計算を共用し、すべての入力が未支持ならinterceptだけで支持を作らない。
headごとの数値errorは他headを無効化しない。全体の観測率は両headの評定失敗時にも独立に保持する。
発行clock、評価clock、4秒horizon、世代の混在を拒否する。対応先以外も含む音群構成変更で凍結混合を無効化する。
身体consumerのwhole-cell publicationへ追加分布を保持し、closure／continuationが欠測でも独立した二headの診断を遮断しない。
従来のclosure／continuation差分が未支持ならその差分はNoneのまま。新しいordinal生成圧力は追加しない。
UIには最新公開候補の評定と支持を折り畳んで表示する。表示コードはbuild済みだが、live画面を実視認したとは扱わない。

独立した確率空間計算で、発行時一致、未来変化、他音群の保持、未知質量、温度とpriorの順序、
全入力欠測、片headのoverflowを検査した。通常Frontend→Gesture／Phrase→候補表→Publicationの固定対応fixtureでは
8評定の変化と7候補のconsumer到達を確認した。音響720 frameとphrase状態は変化しない。
保持される非対応音群の退役、headなしへの変更、古いclock・世代・horizonでも古い混合を使わないことを検査した。

通常releaseは成功一回、stack overflowによる失敗一回、GDB再現一回。最初のGDB起動はsandboxのptrace制約で起動できず、
同じoffline childだけを追跡できる実行文脈で再現した。観測loopからFrontend／Generator／Trackerへ入る時点で上限を超えた。
候補表constructorのinline抑止だけでは領域は減らず、保持snapshotを構築時一回のBoxへ移して解消した。
観測loopの固定stack予約は1,142,216から1,084,648 byteへ減少した。callee領域や保存registerを含む全stack値ではない。
stack上限を変更せず、毎回の候補構築で追加Boxを作らない。失敗版のbinary・source hash・テスト結果・取得ログも保存した。

成功実音では公開434候補の入力・局所ordinal・混合・温度・priorを独立Python計算と照合した。
支持あり評定は両bus二headで856件、候補由来入力を持つcellは248件。最終分布の最大差は2.23e-16未満。
434件は新しい評定が存在する公開cellであり、全内部cell数ではない。通常実身体のpaired consumer正例は0件で、
consumer正例7候補は上記固定対応fixtureに限る。未fit係数による数値照合は実身体転用や聴取者の予測精度を証明しない。

音声・実学習は前版と一致。318候補recordの既存の帰結値と332観測の既存特徴値も一致した。
音群構成変更時に、旧版で残っていた0 cell・対応音群なしの空表が68観測で撤回された。対応する96 packet originと
192件の未成立理由がno_tableへ変わったことを明示して検査し、数値変化としては扱わない。
最大追加invalidationsは4、deferred_callsは176であり、これらを同一だったとは報告しない。
候補表とscratchは最大11,258,096 byte／bus、compact publicationは1,434,272 byte／busとなる。
旧shared-consumer記録の1 MiB未満という保持量は旧版の結果であり、現在版の主張には使わない。全負荷受入は別工程。

全Rust 1,258成功・失敗0・38 ignored、fmt/clippy/all-targets/release通過。
cargo test exit=0 @ 2026-09-19T22:36:57+09:00。
証拠は`target/i10-candidate-groove-heads-20260919/summary.json`、SHA256 `59f016f28a930cc68d7d7589426862d81359357a6c83eaeb9b533badf9551673`。

候補groove／参加意欲の基礎評定の接続・数値検査を閉じる。残り4大工程は共同推論・不確実性・関係支持、
実身体転用と通常consumer正例、残る実default対応、統合検査。共同推論の工程帰属の未回答は維持し、要件を削除しない。
I10全体は未完了。I9保留、I11無効、at-action評価は変更しない。R2全負荷・A3実機受入は別工程。


## sectionの共同遷移adapterと展開時の境界を検査（2026-09-19）

中断されていた登録済み単位を完了した。既存の82座標section headから、stay／new／return／contrastの
四つの既知候補とunknownを作る数値adapterを追加した。子の経路IDは呼出側が渡す。
選ばれた帰還の支持量をraw scoreへ掛け、採用した既知候補を正規化した後、120秒尺度のunknown移行を一度適用する。
帰還先が現在の文脈ならdevelopmentとしてsection境界を維持する。ただし別の子経路であり、
完全な状態維持を表すstayフラグは立てない。従来の双方向の同一視を除き、完全stayと最良の境界組合せを別に予約する。
支持量は候補間の相対重みに反映される。失われたraw scoreをそのままunknownへ足す方式ではない。

定数hazardとsoftmaxの独立した閉形式54条件で、部分支持・帰還先・欠測値・時刻区間を検査した。
経過時間入力の欠測、数値overflowはunknownのまま保持し、不正値・clock・重複IDを拒否する。
整数sample差に基づくunknown移行、完全stayからの境界退出拒否も検査した。
既存sectionの実遷移を呼ぶ検査では、帰還と同一文脈内の展開の二条件でraw scoreとの一致、
展開時のcontext／開始時刻／親経路の保持と、新しい子経路IDを確認した。
adapterから組合せ選択・Normalizerまで通し、完全stayと四つの境界組合せの予約、確率質量保存を二条件で確認した。
これは独立した周辺分布を掛け合わせた通常共同推論ではなく、明示入力による数値部品の検査である。

全Rust 1,260成功・失敗0・38 ignored。fmt、通常Clippy、全target check、release build通過。
cargo test exit=0 @ 2026-09-19T22:59:21+09:00。
release binaryのSHA256は変更前後で `34e960c89f8fc4bc46abd92e806e87b060ffcbf15b847e18c0b377545d7b7761` と一致した。
変更はtest build内の共同推論部品と検査のみであり、通常runtimeの動作変更を主張しない。
記録は`target/i10-section-joint-transitions-20260919/summary.json`、SHA256 `ddcdf9c0b33ce0fdebc80403dbdcc84036993244ac431219f903ac06c9df39aa`。
通常共同推論の状態所有・条件付き実特徴score・全帰結への接続は未完了である。

## 実身体consumerへのgroove／参加意欲評定の到達（2026-09-19）

前段では通常実音の候補評定と実身体consumerの正例が別素材だったため、既存の110 Hz持続音・固定seed・
元のprototype取得recipeで両者をつないだ。距離閾値・prototype・profile・head係数は変更せず、
head無効／有効／temporal offの三実行を取得した。最新release binaryとの同一性も確認した。

候補recordは37件。両bus合計128組でcandidateと実defaultへ二つのheadが届き、各headの116組で両側に支持があった。
対応をfixtureで強制せず、34件の身体bindingの距離・座標数・owner・世代・bus・routing・支持時刻・モデル同一性を照合した。
128組のcandidate／default評定は発行時の凍結混合を使った独立計算と一致し、分布の最大差は1.39e-16未満。
公開された30 cellでは54生入力から109座標・局所評定まで再計算した。

音声は三実行で完全一致。head追加前後の実学習、身体記述、私有trace、候補の身体energyと私有値も一致した。
head有効時に音群構成変更で空表が撤回され、4対応の未成立理由だけがno_groupからno_tableへ変わった。
残るno_body_binding 4件、no_group 12件を正例へ混ぜない。
候補とdefaultのgroove／参加意欲の評定差は全てゼロだった。at-action評価を維持した結果であり、
非ゼロの候補効果や未知の身体への転用精度は証明していない。係数は未fitの診断値であり聴取予測の採用証拠でもない。

記録は`target/i10-actual-consumer-heads-20260919/summary.json`、SHA256 `b06d70776bd4238b43e7716de65319adea9f6e69a55c7ab88c2a742eec15d09a`。
通常consumer到達の正例を追加したが、残り4大工程は共同推論・不確実性・関係支持、実身体転用、
残る実default対応、統合検査のまま。I10未完了、I9保留、I11無効。共同推論の工程帰属に関する未回答は保持する。
R2全負荷・A3実機受入は別工程である。次の一単位は、残る実身体転用の誤差を実recipe／control／既存Toneの対応で切り分け、
共有profileから返す値と実身体予測から返す値の接続条件を確定すること。


## 実身体の振幅平滑化を候補・defaultのenergy予測へ接続（2026-09-20）

I10完了のgoalに従い、既存の転用反例を再計算するだけでなく、実身体条件付きの予測で欠けていた
振幅controllerを接続した。発行時のcurrent／target／丸め済み一sample係数alphaを凍結し、
対象sample自身の平滑化stepを含む閉形式で振幅を評価する。開始振幅が0でも比率で割らず、絶対振幅を予測する。
一定振幅では従来の値を保持し、予約された次のToneUpdate以後は引き続き未知にする。
予測はToneやrendererを進めず、候補から学習しない。追加heap割当はなく、平滑化中だけpowf評価を加える。

指令のsource-energy priorと通常candidate workerの両方が同じ計算を使う。
振幅平滑化だけではstationary sineの位相予測を拒否しない。音高平滑化と予約controlのあるsineは従来の未支持を維持する。
通常候補のDefaultScheduleにはamplitude_smoothing_tonesを独立に記録し、実身体energyの予測が可能でも
未条件付けの共有prototypeを使える根拠にはしない。旧unsupported_control_tonesとの意味を分離した。
model名はsource_energy_log1p_residual_v6、候補footprintはfrozen_body_envelope_sine_bins_v2へ更新した。
既存のPython検証器にも同じ条件の独立した凸結合計算と新版識別を追加した。

8 kHz／48 kHz、0／0.001／0.05／1秒の平滑化、増幅／減衰／ゼロを含む実f32更新との照合で、
閉形式振幅の最大絶対差は3.5762787e-05。実Toneのseeded sine波形では0.001／0.05／1秒・4秒先までを検査し、
最大絶対差は0.00055277663だった。後者にはcarrierの閉形式近似も含む。bit一致や校正済みの精度は主張しない。
通常rendererからcandidate workerまでの検査を拡張し、平滑化中の残存toneを含む候補が既知になること、
両busの音声と、候補診断の有無による実予測学習・私有traceの一致を確認した。

通常releaseの新規実行は一定振幅／振幅変更／時間認知offの3件。
振幅変更素材の318候補record中300件が平滑化を含み、3,976窓が未支持から支持ありへ変わった。
支持を失った窓は0。一定振幅37 recordの既存予測はmodel名と追加ゼロ計器を除いて一致した。
音声は両素材とも旧版と一致し、振幅変更のobserve／offも一致。実観測、descriptor予測、私有trace、
共通52／332観測の受動認知、共有prototypeの対応・拒否は不変だった。
指令source-energyの未来教師は保持し、その予測値・残差学習は変更対象として別に扱った。
この短い比較を実時間性能や身体転用精度の受入には使わない。

全Rust 1,262成功・失敗0・38 ignored。fmt、通常Clippy、全target check、release build通過。
cargo test exit=0 @ 2026-09-20T09:24:04+09:00。
証拠は`target/i10-amplitude-smoothing-20260920/summary.json`、SHA256 `fedd0e1346b44b00eb6c41cc42f827e4de4e640ccc7d1362b3679a0026ae2407`。

I10全体は未完了。共同推論・不確実性・関係支持、実身体転用、残る実default対応、統合検査の範囲を保持する。
次は発行時に既知の予約control更新を、その時刻以後の実身体／default予測へ反映する経路。
未発行の将来方策と混ぜず、実行順・平滑化・支持境界を検査する。I9保留、I11無効、at-action評価維持。R2／A3は別工程。


## 既知の予約振幅更新を実身体／default予測へ接続（2026-09-20）

振幅平滑化に続き、発行時にToneへ予約済みの振幅更新を凍結する経路を実装した。
各Toneの先頭4件までを走査し、振幅の時刻・目標値を固定長で保持する。未対応の音高・駆動更新、
または5件目の時刻を排他的支持境界にする。上限は配列の長さであり、rendererの予約queueを制限しない。
非有限振幅はrendererと同じく無視し、負の値は0へ丸める。同時刻の更新順を維持し、
そのsampleの平滑化より前に最後の目標値を適用する。過去時刻で予約された更新は次の実行sampleへ揃える。

ToneEnergyは元のhold終了、予約release、候補releaseから予約消去の最初の実行sampleを求める。
そのsampleちょうどの更新は適用し、それより後の更新と未支持境界を取り除く。
releaseの物理時刻と実行時刻を分離し、過去向きのreleaseを後から実行しても、
既に適用される更新を過去へ遡って消さない。queued音の発音前も振幅平滑化は進む。
遅延onset候補では既存の絶対更新時刻を保持する。未発行の将来方策を追加した予測ではない。

stationary sineのcarrierは振幅のみの予約では拒否しない。未対応controlまたは上限超過でenergyが未知なら、
coherent計算も未知を保持する。DefaultScheduleのscheduled_amplitude_tonesを追加し、
実身体energyが支持されても共有prototypeの適格性を拡張しない。平滑化中の計数は現在値と目標値の不一致で判定する。
モデルはsource_energy_log1p_residual_v7／frozen_body_envelope_sine_bins_v3へ更新した。
Pythonの独立検証器も区分的な凸結合とreleaseによる予約消去へ対応した。

8／48 kHz、平滑化0／0.001／0.05／1秒と丸めでalpha=0になる条件、発音済み／queued、
同時刻・過去時刻・release前後の予約を含む120条件で実Toneと照合した。
振幅の最大絶対差4.2757629e-07、波形2.0853188e-05。
予測の有無による実音はcloneと一致した。上限超過・音高・駆動更新の支持境界も検査した。
Pythonの独立765点では反復f32更新との差1.1308295e-07。
型サイズはControlForecast 176 bytes、ToneEnergy 304 bytesで、それぞれ80 bytes増えた。
snapshot経路に新しいheap割当は追加していないが、これはR2全負荷検査の代わりにはならない。

通常releaseで一定振幅／振幅変更の2素材を再実行した。振幅変更318 recordのうち16 recordが予約振幅更新を含み、
224窓が未支持から支持へ変わった。支持喪失0、以前から支持されていた窓の値変更0。支持窓の総数は5,088。
一定振幅37 record・592窓は識別子と新設ゼロ計器を除いて一致した。
両素材の音声、実観測、非energy学習、私有trace、共通52／332件の受動認知、共有profile対応・拒否は旧版と一致した。
通常候補workerの検査にはqueued、後続batch、複合defaultを含む。旧版の「振幅予約も未支持」という4期待値は更新し、
未対応controlの検査は音高更新ケースを追加して保持した。初回失敗ログも保存した。

全Rust 1,264成功・失敗0・38 ignored。fmt、通常Clippy、全target check、release build通過。
cargo test exit=0 @ 2026-09-20T09:42:19+09:00。
証拠は`target/i10-scheduled-amplitude-20260920/summary.json`、SHA256 `5ff491880d83c7adbbc87deeac13baa3d72b4f4252ad802a3858bf783bac94f8`。

I10全体は未完了。共同推論・不確実性・関係支持、実身体転用、残る実default対応、統合検査の範囲を保持する。
予約control対応は振幅の先頭4件に限られ、音高・駆動および未発行の将来方策は残る。
次は通常共同推論に必要な状態所有と実条件特徴の接続を確認し、実身体転用/defaultの残差と統合する。
工程帰属に関する未回答は保持し、未接続の列を削除して完了とはしない。I9保留、I11無効、at-action評価維持。R2／A3は別工程。


## 共同経路の前回重みと親identityを保持する層（2026-09-20）

通常共同推論に必要な前回重みと親経路の所有を、既存の共同正規化核へ追加した。
Stateはbus／epochに属し、観測区間を一度だけ進める。観測ありはcanonical hop、欠測は連続したhop境界区間に限る。
呼出し側は遷移と観測scoreを渡すが、priorは渡さない。保持中の共有／局所重みから内部で取得し、
全列挙の正規化後に次の有界inventoryを構築する。unknown priorは明示的unknownと枝刈り質量の和であり、
破棄したidentityをknown parentとして復活させない。共有unknownには局所known parentを保持しない。

共有・局所pathにはbus／epoch／単調IDのhandleと直前の親を付ける。現在のcontext keyやcomponent keyが同じでも、
別の親経路を統合しない。局所親は対応する共有親と音群handleの下だけで有効とする。
音群の並び替えでは正しい前回位置を使い、退役後の同一世代再投入を拒否する。新世代はunknown priorから始める。
重複／飛越／逆行区間、他bus／epoch、未保持の親、親の列挙欠落、正規化不正、ID枯渇を拒否し、
公開済みsnapshot・時計・次IDは変えない。失敗した入力の後にも有効な再試行ができる。

独立した確率空間の計算と10更新を照合した。最大重み差1.6653345e-16。
観測／欠測、同じcontext keyを持つ別親、音群順の反転、局所／共有枝刈り、全音群退役と新世代を含む。
32共有候補・8音群・57,856局所候補の容量条件でも、unknownと枝刈り質量、安定した上位経路を保持した。
共有unknownは局所親を1個だけ持つため、この条件の実列挙数は保守的上限65,536より少ない。
65,536 Pairのscratchと137,976-byte snapshot二つを初期化時に確保し、更新では再利用する。
このbyte計数とpointer再利用は全workerの時間・memory受入を代用しない。

全Rust 1,268成功・失敗0・38 ignored。fmt、通常Clippy、全target check、release build通過。
cargo test exit=0 @ 2026-09-20T09:57:45+09:00。
追加実行した全target Clippyは既存6ファイルの18指摘で失敗した。6ファイルが前単位から不変であることをhashで確認し、
今回追加したownerへの指摘は0だった。必須の通常Clippy通過と追加検査の失敗を区別して保存した。
証拠は`target/i10-joint-owner-20260920/summary.json`、SHA256 `4ce1891159712c0515876af5659e9c5615cb51d2208310d597b97a378b7040de`。

joint全体は引き続きcfg(test)であり、通常workerへの接続は未完了。release binaryは前単位とSHA256まで一致した。
今回所有したものはjointの質量と直前親identityである。component keyの実状態・履歴・支持、
commit窓のpayload所有、共有proposalの実選択を実装済みとは扱わない。
次は実component状態のproducerと、このownerを使うproposal組立を接続し、登録した条件付き特徴からscoreを供給する。
その後に通常共同推論・候補帰結へ通す。I10の残る共同推論・実身体転用・default・統合検査を維持し、完了とはしない。
I9保留、I11無効、at-action評価維持。R2／A3は別工程。


## 実観測のarticulation状態を共同経路へ保持（2026-09-20）

通常Gestureと共同proposalが使う実観測入力・状態遷移を共通化した。
入力は元のRawDescriptorの音群handle、区間、source／available時刻、既知sample数と有限値を検査し、
既存のrise／flux／motion等11座標を同じ順序で組み立てる。遷移rateの経過時間は選択された共同親のenteredから計算する。
独立Gestureの周辺重みを共同経路のpriorへ流用しない。将来の親時刻・観測端点は拒否する。

共同ownerはarticulationのラベルだけでなく、entered、attack／release／gapの元観測支持、
最後の観測端点と欠測によるcensoringを保持する。次の観測のproposalはこの実状態から作る。
通常Gestureも同じ入力と状態更新を使うが、質量計算・枝刈り順は維持した。
他4 componentの履歴、commit窓のpayload、実共有proposal選択と全条件付きscoreを完成したとは扱わない。

独立した確率式60条件で観測／欠測／静音、親4状態とunknown、経過時間、非ゼロ係数と標準化を照合した。
6回の実RawDescriptor入力をproposal→owner→次入力へ渡し、保持状態203件、stay時の開始時刻97件を検査した。
release→gapの元観測支持も保持し、欠測frameを観測端点にしない。
初回の既定ゼロ係数fixtureでは検査対象のrelease→gap履歴が枝刈りされたため、
テスト入力のrelease／gap biasを20へ設定して対象履歴を保持した。実行時の既定値や閾値は変更していない。初回ログを保存した。
snapshotは285,816 bytesへ増え、二つを初期化時に確保して再利用する。新しいhop内heap割当はない。

通常releaseで一定振幅／振幅変更／非ゼロ係数のpulseを旧版と比較した。
393候補record・434件の受動認知観測、発行済み予測と実結果、私有trace、共有profile対応は、処理時間計器を除いて一致した。
3素材のWAVもSHA256まで一致した。pulseの係数は分岐を通す工学検査用であり、学習・採用結果ではない。
全Rust 1,271成功・失敗0・38 ignored。fmt、通常Clippy、全target check、release build通過。
cargo test exit=0 @ 2026-09-20T10:12:00+09:00。
証拠は`target/i10-joint-articulation-20260920/summary.json`、SHA256 `8e6ae325de6226c90b9c40a1a5a752c654ca76df7c00dfb3b72fce27bb2ed1d1`。

joint全体は引き続きcfg(test)。通常共同推論への接続は未完了。
次は残るcomponent状態のproducerと履歴保持、共有選択と条件付きscoreを接続する。
I10の共同推論・実身体転用・残るdefault・統合検査を維持し、完了とはしない。
I9保留、I11無効、at-action評価維持。R2／A3は別工程。


## phraseの実履歴を共同経路へ保持（2026-09-20）

通常phraseのforeground、終了descriptor、継続linkの更新を、重みから独立したInterpretationへ分離した。
通常版の質量計算・sort・枝刈り順は維持し、jointも同じ遷移処理を使う。
状態は開始・最終聴取時刻、credit、再解釈・終了event、直前に閉じたforegroundとdescriptor、
最大16件の継続linkと容量喪失数を保持する。再解釈は新しいidentityを持つが開始・creditを保ち、
overlapは元の聴取端点をright-censored linkへ残す。欠測stayは聴取端点を進めない。

jointの候補は親の保持slot・遷移種・identityだけを持つ。親slotが選択された局所経路と一致するか、
時刻とidentityが有効かをownerが検査し、親の実状態を内部から取り出す。
呼出し側の差し替えpayloadやsnapshotの借用を必要としない。
全候補を正規化して枝刈りした後だけ、保持する履歴を次のarenaへ実体化する。
active phraseのproposalは保持されたforeground開始から経過時間を算出し、選択された一つのarticulationを条件にする。
独立phraseの周辺重みを共同priorとして渡してはいない。

全4 exit、元descriptor、再解釈credit、欠測と16-link容量喪失を検査した。
共同ownerを6更新し、162件の保持履歴を次の条件付き遷移へ渡した。
他経路slot・範囲外slot・identity不一致・支持の未来時刻を拒否し、snapshotと次IDを保ったまま有効な再試行が成功した。
初回の再試行fixtureは共有親のunknown枠が欠けていたため、枠と正規化遷移を修正した。
拒否検査は期待する理由も照合する。初回ログを保存した。

Interpretationは1,280 bytes、候補の変更指示は32 bytes。
840枠・1,075,200 bytesのarenaを二つ、初期化時にheap確保して再利用する。
snapshot本体は299,272 bytesであり、arenaのheap領域を含まない。
候補ごとの大きな履歴複製と新たなhop内heap割当はないが、全workerの時間・memory受入は別途必要である。

通常releaseの一定振幅／振幅変更／非ゼロ係数pulseを旧版と比較し、393候補record・434観測、
発行済み予測・実結果・私有trace・共有profile対応が処理時間計器を除いて一致した。
3素材のWAVもSHA256まで一致した。pulseの係数は工学検査用であり、学習・採用結果ではない。
全Rust 1,274成功・失敗0・38 ignored。fmt、通常Clippy、全target check、release build通過。
cargo test exit=0 @ 2026-09-20T10:31:29+09:00。
証拠は`target/i10-joint-phrase-state-20260920/summary.json`、SHA256 `acb10baab06523bdb05e1a67083d0b5df26917b508c6733132206be92c4bc104`。

jointとconsequenceは引き続きcfg(test)。この単位は実状態の共通更新と共同保持までであり、
通常の共同推論、実観測からの全phrase特徴・終了descriptor更新、unknown／inactiveの実音admissionを完成したとはしない。
6更新の検査はactive遷移の実装を使うが、観測支持とadmission priorはfixtureである。
次は残るcomponentの実状態・共有選択・条件付き特徴とscoreを接続する。
I10の共同推論・実身体転用・残るdefault・統合検査を維持する。I9保留、I11無効、at-action評価維持。R2／A3は別工程。


## 実matcherの対応候補と探索範囲を共同proposalへ接続（2026-09-20）

受信済みのResultSnapshotと全変換候補から、既存の14座標音響scoreを用いて共同correspondenceの候補を作る入口を追加した。
元のquery／episode世代、cue開始とoccurrence、音響・可用・発行・完了・受信時刻、変換・残差・圧縮誤差と検索の枝刈りを保持する。
同じqueryの再読でidentityを増やさず、同じ対応の新しいqueryは親component IDを保って子の支持だけを更新する。
親payloadそのものは変更しない。現在のepisode世代一覧から退役対象を除外し、追加DTWを実行しない。

新規候補は全受信変換の音響score順に最大16件。tieはepisode／世代／anchor／変換で固定する。
16 episodeの全変換を無制限に保持する規則ではなく、変換候補自体の上限16である。除外件数を明示する。
保持中のstayはこの16枠の外に置き、同じcue開始／occurrenceと同じepisode／世代／anchor／変換を重複させない。
既知の「記憶対応なし」はSome(0)、認識的unknownはNoneであり、共同正規化でも混ぜない。
既存の10秒unknown leakと最大19枠、16 tuple上限へ通す。raw scoreはexp(log_score-max)を既存Listで正規化する。
新規係数をfitした結果ではなく、明示的に渡された既存係数とno-memory biasによる遷移候補である。

検索の不足を対応なしに変換しないため、通常matcherのsearch_coveredを追加した。
候補枝刈り／cutoff tieがなく、全eligible coarse候補のcostが支持され近似旗がなく、各対象に支持された非band-edge refinementがある場合だけtrueとする。
既存のmatching loopと一度のcoarse走査で集計し、再matchingや新しいheap割当は加えない。
空のeligible bankは範囲内の探索完了だが、非空検索による新規性の証明とは区別する。
通常queryへ公開し、sectionの固定search_covered=trueも元queryの値へ置き換えた。
新規no-memory候補はこの証拠と明示bias、観測あり、0.5秒未満の元音響支持、deadline内という条件を要する。
欠測／期限切れの未知親からknownを復活させず、既存stayの古い音響特徴はmaskする。

合成raw入力を実Recall／matcherへ72 hop通し、208候補・55 queryの元支持と全変換payloadを照合した。
既知no-memory候補は63件。別の実matcher条件で、空の記憶32件、容量不足47件、変換範囲外16件を区別し、
不十分な探索から新規no-memoryを出さないことを確認した。16 fresh＋cached stay＋known no-memory＋unknownの19枠、
64変換からの48件除外、入力順不変、退役・欠測・期限・ID枯渇時の拒否も検査した。
対応候補を既存ComposerとNormalizerへ通し、known no-memoryとall-unknownの区別、遷移質量を独立した確率式と照合した。
cacheは12,192 bytes、状態472 bytes、出力9,160 bytes。全workerの時間・memory受入ではない。

通常3素材の1,982 distinct queryで新しい探索範囲flagを確認した。trueは498件、falseは1,484件。
追加したquery fieldと処理時間計器だけを比較対象から除き、393候補record・434観測、発行済み予測・実結果・私有trace・共有profile対応は旧版と一致した。
3素材のWAVもSHA256まで一致した。
全Rust 1,280成功・失敗0・38 ignored。fmt、通常Clippy、全target check、release build通過。
cargo test exit=0 @ 2026-09-20T10:52:27+09:00。
証拠は`target/i10-joint-correspondence-20260920/summary.json`、SHA256 `db1472d68fe9e957a7f508c04fe3d98932563ded1ad391147bb2e72f5077a4a8`。

correspondenceのCache／Listとjoint全体は引き続きcfg(test)。通常接続されたのは元queryの探索範囲とsectionへの伝達である。
次は対応payloadのjoint ownerへの保持、grouping／section／共有proposalの実状態、全条件付き特徴とscoreを接続する。
実matcher検査は合成raw入力であり、実音の対応精度や人の認知的受入を証明しない。
I10の共同推論・実身体転用・残るdefault・統合検査を維持する。I9保留、I11無効、at-action評価維持。R2／A3は別工程。


## 対応payloadを共同ownerで保持（2026-09-20）

共同correspondenceの候補を、現在のquery cacheのslotまたは選択された親経路を指す小さな指示へ変更した。
ownerはcomponent ID、group／epoch、cutと元支持時刻、episode世代の現存を検査する。
保持親のpayloadはowner自身から取り出し、呼出し側が別経路のpayloadへ差し替える入口を設けない。
同じ対応の新queryは親component IDを保ちながら子の支持を更新する。元の親payloadは不変である。
既知no-memoryをcacheの専用slotに置き、Some(0)と認識的Noneの区別を維持した。

全列挙候補の正規化と枝刈りの後、保持された状態だけを再利用arenaへ複製する。
840枠・396,480 bytesのcorrespondence arenaを二つ用意し、更新間で再利用する。
snapshot本体319,448 bytesにはこのheap領域も既存phrase arenaも含まない。
候補の変更指示は16 bytes、cacheは12,664 bytes、状態472 bytes、出力9,192 bytesである。
新たなhop内heap割当はないが、全workerの時間・memory受入を通したとはしない。

合成raw音響特徴を実Recall／matcherへ96更新通し、候補→Composer→owner→次queryを検査した。
保持経路における対応あり6,268件、対応なし817件、元payloadの継承1,187件を照合した。
これらは複数更新・複数経路にまたがる照合回数であり、独立した音響イベントの件数ではない。
退役対象を持つ親のproposal呼出し300件でその対応を除外した。300個のepisodeを退役させたという意味ではない。
欠測中は元の支持を保ち、音響特徴をmaskする。参照元欠落の95回の拒否でもsnapshotと次IDを保ち、その後の有効な再試行が成功した。
別検査で未来cut、偽ID、親欠落、範囲外slot、他groupと退役対象を拒否した。

通常3素材の393候補record・434観測、発行済み予測・実結果・私有trace・共有profile対応は、処理時間計器を除いて旧版と一致した。
前単位で追加したsearch_coveredも今回は比較対象に含めた。WAVも3素材すべてSHA256まで一致した。
全Rust 1,282成功・失敗0・38 ignored。fmt、通常Clippy、全target check、release build通過。
cargo test exit=0 @ 2026-09-20T11:06:11+09:00。
証拠は`target/i10-joint-correspondence-owner-20260920/summary.json`、SHA256 `3cc587389f2e739a7d4044cf2f7041b38fc90920a5fa4e6fceaaf86c7b2def97`。

joint全体はまだcfg(test)。この検査の共有contextと他componentは制御したfixtureであり、通常の全共同推論や実音の対応精度を検証した結果ではない。
次はgrouping／section／共有proposalの実状態、全条件付き特徴とscore、通常接続を進める。
I10の共同推論・実身体転用・残るdefault・統合検査を維持する。I9保留、I11無効、at-action評価維持。R2／A3は別工程。


## 実grouping inventoryを共同proposalへ接続（2026-09-20）

既存のperiod Estimatorとgrouping Inventoryから、共同groupingの候補を作る入口を追加した。
period bin、integer／ordered-word形状、全accent anchorと個数を含む元keyで同一性を判定する。
同じinventoryの再読ではIDを増やさず、名称だけが同じ候補をまとめない。
元のProposalを保持するため、二反復のduration・coverage、timing residual、source／available時刻、
skipped accentと観測pairの旗を、短い診断snapshotから推測し直さない。追加のaccent走査やperiod計算は行わない。

既存のendpoint-weightを正のraw admission scoreとし、最大値で共通に割ってから既存の10秒unknown leakへ渡す。
新規16候補の外に保持stayを置き、同一keyを重複させない。既知の群化なしはSome(0)、認識的unknownはNoneである。
群化なしの新規admissionには明示した正のprior、元period支持の一致、window・capacity・候補除外の不足がないという条件を課す。
これは探索条件付きの解釈候補であり、新規性の証明ではない。除外件数はadmission case数であり、uniqueな解釈数ではない。
元period／accent支持が一致しないcacheや欠測入力は新規knownを作らない。
保持stayは古い支持を元のまま持ち、現在の観測inventoryに同一keyがない場合は特徴をmaskする。

合成Accent入力を実Estimator／Inventoryへ90更新通し、1,231件の候補payloadと89件の継承を検査した。
各候補の遷移質量を独立した確率式と照合し、最大誤差は5.551e-17であった。
この検査はgroupingありのpayloadを主に照合し、早期の群化なしadmissionを数えていなかった。単一accentからのadmissionの不備は後続の共同owner検査で判明し、下記の修正で対応した。
最大19枠の別fixtureは探索完了flagを明示的に制御しており、16 fresh＋stay＋群化なし＋unknownから16 tupleへの制限を検査する。
window不足、capacity喪失、候補除外それぞれで新規の群化なしを止め、保持済みの群化なしは古い支持を保ってmaskする。
別group、時刻逆行、ID枯渇の拒否と有効な再試行、空inventoryと欠測からの不当なknown復帰がないことも検査した。
cacheは11,344 bytes、状態664 bytes、出力12,608 bytes。全workerの資源受入ではない。

通常groupingファイルはtemporal module内の可視性とPartialEq導出だけを変更した。
その差を除けば元sourceと一致することを記録scriptで確認した。新規の実行処理はすべてcfg(test)内にある。
この単位では通常renderを反復しない。前単位の3素材一致とは区別する。
全Rust 1,285成功・失敗0・38 ignored。fmt、通常Clippy、全target check、release build通過。
cargo test exit=0 @ 2026-09-20T11:16:50+09:00。
証拠は`target/i10-joint-grouping-20260920/summary.json`、SHA256 `184940cddd098a16f03d89f138e139a863ef9fd17e85dbb9e284fb2c4ca61994`。

groupingの共同owner保持、section／共有状態、全条件付き特徴とscore、通常接続は引き続き必要である。
この検査は実音入力や人のgrouping判断、全特徴登録・fitの結果ではない。
I10の共同推論・実身体転用・残るdefault・統合検査を維持する。I9保留、I11無効、at-action評価維持。R2／A3は別工程。


## groupingの共同保持と早期admissionの修正（2026-09-20）

共同ownerがgroupingの元key・残差・支持区間を保持し、次の観測では選択された親経路からそのpayloadを取り出すようにした。
候補は現在のcache slotまたは保持親を指す16-byteの指示だけを持つ。参照元・group／epoch・cut・元時刻・component IDを検査し、
全候補の正規化と枝刈りの後だけ保持payloadを次のarenaへ複製する。group配列のslot番号を恒久identityとして扱わない。
correspondenceとgroupingの参照元をSourcesへまとめ、correspondence側の親参照も同じ選択済み経路から解決する。

実Estimator／InventoryからComposerとownerを90更新通し、8,930件の保持payload、420件の元payload継承、89回のgroup順序変更を検査した。
参照元欠落の86回の拒否でもsnapshotと次IDは不変で、有効な再試行は成功した。
欠測では音響特徴をmaskし、古い支持を更新しない。別group、偽ID、誤ったcut、範囲外slot、保持親欠落も拒否する。
これらの件数は保持経路に対する照合回数であり、独立した音響イベントの数ではない。

初回検査で、前単位のCacheが単一accentから既知の群化なしをadmitしていたことが判明した。
元accent支持が非空でもperiod探索を支持するpairがなければ、探索完了とはしない。
新規の群化なしには、従来の条件に加えて支持されたperiod pairと一つ以上のperiod peakを必要とし、元の件数もSupportに保持した。
単一accentからのknown復帰を止める回帰検査を追加した。前単位の90更新はgroupingありの候補を主に検査しており、この早期admissionは未検出だった。
また、古い正規化fixtureのpayloadを持たないgrouping IDを取り除いた。確率参照式は変更せず、10更新の最大誤差は1.665e-16のままである。
初回失敗ログも保存した。

修正後のproducerは1,231件の候補を照合し、遷移質量の最大誤差6.939e-17。
grouping状態は680 bytes、cacheは11,616 bytes、出力は12,928 bytes。
ownerのgrouping arenaは840枠・571,200 bytesを二つ再利用する。snapshot本体332,904 bytesにはheap arenaを含まない。
共同推論関連41テスト通過。全Rust 1,287成功・失敗0・38 ignored。
fmt、通常Clippy、全target check、release build通過。cargo test exit=0 @ 2026-09-20T11:25:50+09:00。
変更source六つはすべてcfg(test)のjoint配下であり、通常renderは反復しない。
証拠は`target/i10-joint-grouping-owner-20260920/summary.json`、SHA256 `0342de33001d73156855c8b13b61d21b10084ea172e1a6b6f31565c52bc18d7d`。

この単位も通常の全共同推論や実音による認知的妥当性の完成ではない。共有contextと他componentは制御したfixtureである。
残るsection／共有状態、全条件付き特徴とscore、通常接続、実身体転用・残るdefault・統合検査を進める。
I10未完了。I9保留、I11無効、at-action評価維持。R2／A3は別工程。


## sectionの解釈状態を通常版と共同proposalで共有（2026-09-20）

sectionの経路を、重み・経路ID・遷移と、音響解釈のInterpretationに分離した。
Interpretationはgroup、元の開始時刻と更新端点、context、累積・直近履歴、元query／focus、82座標、
最大512件の終了phrase所有記録、遅延accent支持を保持する。重みを持つ独立sectionの診断snapshotから再構成しない。
通常formのstay・新context・return・contrast・developmentは同じInterpretation::advanceを呼ぶ。
既存のcompletion割当、search_covered、context reset、遅延accent処理の算術と順序を維持した。

共同section proposalは保持されたstart／end、contextと82座標を直接参照して経過時間と遷移質量を計算する。
現在と同じcontextへのreturnはdevelopmentとしてstay境界、異なるcontextはexit境界とする。
候補contextのadmission支持と経路IDはまだ呼出し側の入力であり、全共同推論や新規fitの完成ではない。

履歴とInterpretationのclone_fromは保持先のVecを再利用し、context resetも直近4件の履歴bufferを再利用する。
通常のcloneは従来の挙動を保つ。新たなhop内heap割当は加えない。
5種類の遷移で元履歴、context開始、関係、bufferのpointer維持を確認した。
欠測は元の観測支持を伸ばさず、更新時刻と欠測割合だけを進める。異なるActivity窓や欠測中のcontext exitを拒否した。
遅延accentは元の終了phraseへ帰属させ、期限内と期限後の支持を分ける。
既存のbyte-exact履歴fixtureも通過した。

共同遷移質量を独立した確率式12条件と照合し、最大誤差は2.220e-16。
Interpretation本体は3,216 bytesであり、4×640-byteの履歴と512×24-byteの所有記録のheap容量は含まない。
この容量確認は全workerの時間・memory受入ではない。
通常releaseの3素材を直前の通常比較版と照合し、393候補record・434観測、sectionを含む全認知出力、
発行済み予測・実結果・私有trace・共有profile対応は処理時間計器を除いて一致した。WAVも3素材すべてSHA256一致。
section関連33テスト、全Rust 1,290成功・失敗0・38 ignored。
fmt、通常Clippy、全target check、release build通過。cargo test exit=0 @ 2026-09-20T11:37:18+09:00。
証拠は`target/i10-joint-section-state-20260920/summary.json`、SHA256 `665ba97a6e010a22183bd1033f286d4c9009b52f6ba79e177f859802f5e9e67d`。

通常接続されたのは共通のsection解釈更新である。joint／consequenceは依然cfg(test)。
次はsectionの共同owner保持、共有状態・全条件付き特徴とscore、通常接続を進める。
I10の実身体転用・残るdefault・統合検査も維持する。I9保留、I11無効、at-action評価維持。R2／A3は別工程。


## section履歴を共同ownerで保持（2026-09-20）

共同ownerがsectionのInterpretationを保持し、次の条件付きproposalがそのstart／context／82座標を読む経路を追加した。
32-byteの指示が選択された親section slotと不変のsource recordを参照する。source自身にも局所親Handleを付け、
他経路の条件付き入力を差し替えない。component ID、親slot、group・区間・観測旗を正規化前に検査する。
新規admissionは最初の区間の左端から観測を取り込み、既存sectionのexitは従来どおり右端でcontextを切り替える。

sectionの履歴更新はエラーを返し得るため、枝刈り後の保持状態だけを未公開の次arenaへ更新する。
途中で失敗しても現在snapshotと次path IDは変更しない。次の有効な試行は実際の保持親から更新先を上書きする。
両arenaを独立に初期化し、空Vecのcloneで予約容量を失わないようにした。
各側840個のInterpretationと、それぞれ4件のHistory、512件の終了phrase所有記録を初期確保する。
生きたsectionはLocalPath.section_slotが参照するものだけであり、未使用slotに残る内容から枝刈り済みidentityを復帰させない。

8更新で248件の保持状態、開始時刻を保つ103件の継続、27件のreturnと5件のdevelopmentを検査した。
参照元欠落と別親sourceを各8回、履歴更新中の非有限値を8回拒否した。
最後のケースは正規化後のpayload更新に到達するが、公開snapshotと次IDは不変で、有効な再試行が成功した。
保持先のHistory・所有記録のpointerと容量は維持された。これらの件数は保持経路への照合回数である。

section arena本体は片側2,701,440 bytes。Historyは片側2,150,400 bytes、所有記録は片側10,321,920 bytes。
両側のsection確保量は計30,347,520 bytesであり、allocator管理領域、他のcomponent、source cacheを含まない。
snapshot本体は346,360 bytesでheap arenaを含まない。全workerの時間・memory受入は未実施である。
共同推論関連42テスト、全Rust 1,291成功・失敗0・38 ignored。
fmt、通常Clippy、全target check、release build通過。cargo test exit=0 @ 2026-09-20T11:50:27+09:00。
通常ファイルの差はPartialEq導出とcfg(test)のreset／storageだけと照合したため、通常renderは反復しない。
証拠は`target/i10-joint-section-owner-20260920/summary.json`、SHA256 `5ea7cadd26399a175069ae224c1ce111ce2b5e138f72baf0c5d37cec530d2c29`。

jointはまだcfg(test)。この検査のActivity・completion・共有context・return選択は制御したfixtureである。
通常の条件付きsource組立、選択されたcorrespondenceと過去contextの証拠照合、unknownから過去contextへのcue駆動復帰は残る。
source record上限65,536は構造上の上限であり、通常producerのcache効率や時間予算の達成を示さない。
共有状態・全条件付き特徴とscore・通常接続、実身体転用・残るdefault・統合検査を維持する。
I10未完了。I9保留、I11無効、at-action評価維持。R2／A3は別工程。


## section復帰を選択済みcorrespondenceと確定contextへ結合（2026-09-20）

共同sectionのreturn／developmentは、選択されたfresh correspondenceと、元episodeの確定済みcontext支持を必要とする。
episode IDだけでなくgenerationも照合する。この世代照合は通常sectionのprovenance参照にも適用した。
元phrase区間はquery cueより前、確定時刻はquery発行以前でなければならない。
queryの元の支持区間、cue identity、受領時刻、deadline、音響支持の鮮度も保持する。
最大16個の固定長cacheは既存matcher結果を参照し、追加DTWを実行しない。

各correspondenceについて、保存された正のcontext支持の最大値を選ぶ。同率は小さいcontext IDを優先する。
残ったcontext間で支持質量を再正規化せず、選択済み対応の音響scoreを二重に掛けない。
これは過去context候補の証拠を供給する処理であり、全条件付きscoreの完成ではない。
共同ownerはcorrespondenceを先に解決し、不変のreturn根拠、query、現在の区間終端、sample rateを照合する。
元のqueryやfocus、contextを差し替えたsourceは拒否する。保持済み対応の継承だけ、欠測、期限外、退役対象からは復帰しない。

unknown sectionも、証拠のある過去contextへ復帰できる。現在区間の左端から観測を取り込み、元episodeの履歴をコピーしない。
同じcontextを保持した既知親ではdevelopmentとなり、元の開始時刻を保つ。
制御したcontext支持の3条件（0.6／0.2、0.4／0.4、0／0）それぞれ100更新で、実際のcue確定・Recall・matcherを通した。
正の2条件は各84窓で復帰支持が得られ、各条件の最初の6窓で共同ownerを検査した。
保持経路の照合はunknown復帰24件、development12件。根拠欠落・query差し替え・別context・focus欠落・退役対象の計60更新を拒否し、公開状態とIDの保全、有効な再試行を確認した。
支持ゼロではmatcherが動いても復帰候補を作らない。同率と未割当質量の保持も確認した。
これらは独立した音響イベント数や学習済み確率の正しさを示す件数ではない。

従来の共同section owner検査から、制御fixtureだけによるreturn／developmentを取り除いた。
同検査は222保持・84開始維持となり、return系は今回の実provenance検査へ移した。
共同推論関連43テスト、全Rust 1,292成功・失敗0・38 ignored。
fmt、通常Clippy、全target check、release build通過。cargo test exit=0 @ 2026-09-20T12:07:52+09:00。
通常releaseの3素材では393候補record・434観測、発行済み予測・実結果・私有trace・sectionを含む認知出力が処理時間計器を除き一致した。
WAVも3素材すべてSHA256一致。証拠は`target/i10-joint-section-returns-20260920/summary.json`、SHA256 `dc6fa5b3f6aa1870737f9a7a29c38f2e8235f1cf6d4b562913b63b360d6349b4`。

joint／consequenceは依然cfg(test)。Activity・phrase・accentの全条件付き入力組立、共有状態・全特徴とscore、通常接続は残る。
実身体転用・残るdefault・統合検査も維持する。I10未完了。
I9保留、I11無効、at-action評価維持。R2／A3は別工程。


## section入力組立とphrase支持の共同保持（2026-09-20）

通常sectionの音響Activity組立とphrase観測蓄積をsection/input.rsへ共通化した。
元の割当支持・物理窓、群化／共存の支持、phrase credit、終了descriptorの標準化、隣接性、accent帰属の算術と順序を維持する。
共同ownerも同じ処理を呼び、grouping occupancyは実際に選択した支持付きgroupingから決める。
既知のungroupedは既知ゼロ、継承のみ／unknownは現在の支持なしとして区別する。
他groupが観測中でも欠測groupの観測log-potentialは0にし、そのgroupのunknownから既知状態を再生成しない。

phraseごとに696-byteのSpanを保持し、sectionがunknownでもphrase支持を失わない。
両arena各840 slotの追加heapは計1,169,280 bytes。選択された親のSpanを固定arena内でコピーし、枝刈り後の子だけを更新する。
生きたSpanはphrase_slotが参照するものだけであり、未使用slotの残存値から復帰しない。
終了記録は選択されたphraseの元credit・区間と蓄積値から生成する。独立phrase posteriorの平均を渡さない。
音響source inventoryを供給した場合は、所有group・区間・sample rateとbase deltaを照合し、外部で組み立てたcompletion／deliveryを拒否する。
既存の部品単体fixtureはinventoryなしの制御入力を使う。通常producerではinventoryを必須にする統合が残る。

12更新のfixtureで実音響入力処理と実grouping inventoryを使い、phrase終了・再解釈・section unknown・group欠測を通した。
622保持経路の割当支持を観測時刻から独立再計算し、完了phraseを持つ287経路、section unknown中の支持245経路を確認した。
所有者不一致12回、正規化後の物理支持エラー10回を拒否し、公開snapshotと次IDの不変性、有効な再試行を検査した。
これは12更新内の保持経路数であり、独立した音響イベント数ではない。

共同関連44テスト、通常section関連33テスト、全Rust 1,293成功・失敗0・38 ignored。
fmt、通常Clippy、全target check、release build通過。cargo test exit=0 @ 2026-09-20T12:21:44+09:00。
通常3素材の393候補record・434観測、認知・学習・発行済み予測とWAVは前版と一致した（処理時間計器を除く）。
証拠は`target/i10-joint-section-inputs-20260920/summary.json`、SHA256 `bbf23766ede0b243d013781809ce7693cc05485665ad7804aa906ebc31af6c86`。

残件を`target/i10-joint-section-inputs-20260920/remaining-audit.json`へ再点検した。
共有状態／全条件付きscore、通常共同推論、全候補帰結、実身体転用、残るdefault、統合・資源引渡しの6実装群を維持する。
既に通常接続されたgroove／desire headを未実装扱いしない。今回もjoint／consequenceはcfg(test)。
作業時間は半日〜2日規模の低確度見積もりであり、完了保証や合格条件ではない。実身体転用の修正量が主な不確定要因である。
I10未完了。I9保留、I11無効、at-action評価維持。R2／A3は別工程。


## 共有経路ごとの前回summaryを保持（2026-09-20）

共同ownerの共有ContextPathはkeyだけでなく、更新終端と8 group分の型付きsummaryを持つStateを保持する。
各groupのarticulation、grouping、phrase活動、correspondence、局所section contextの質量を、保持された局所経路から集計する。
質量はq(local|shared)であり、共有経路のq(shared)を再び掛けない。既知のungrouped／inactive／no-memoryをunknownから区別する。
全局所unknownと枝刈り質量も別に保持し、既知質量を再正規化しない。
同じ局所section contextの集計はsummary上の射影だけであり、経路や履歴を統合しない。

このsummaryは、共同正規化と保持payload更新の後に一度だけ作り、arena交換時に公開する。
同じkeyを持つ共有経路も別Stateを保持する。次の特徴組立は前回snapshotを参照し、今回の事後分布を同じ更新の追加観測として再投入しない。
元の音響interval・source開始・availability・割当秒数は、別のsupport欄に保存する。
欠測ではcurrent supportを空にし、最後の観測終端を元group Handleにだけ継承する。slot移動や新generationは観測支持を引き継がない。
component入力しかないfixtureでは、observed旗だけから音響支持を作らない。

同じkeyを持つ2共有経路・2 groupの3更新について、48個のarticulation質量を独立した確率計算と照合した。
group順序変更、groupごとの欠測、退役slotへの別generation追加、元の観測時刻を検査した。
実matcher・grouping inventory・phrase蓄積を通す既存fixtureにも、既知ゼロとunknownの区別、条件付き総質量、section label射影の検査を加えた。
Stateは4,432 bytes、Groupは552 bytes。snapshot本体は377,344 bytes（heap除外）。
旧key分を差し引く両arenaの増分は61,936 bytes。固定配列であり、hopごとのheap確保を追加しない。全worker資源受入の証拠ではない。

共同関連45テスト、全Rust 1,294成功・失敗0・38 ignored。
fmt、通常Clippy、全target check、release build通過。cargo test exit=0 @ 2026-09-20T12:34:37+09:00。
変更した8 sourceはすべて、変更していないcfg(test) joint配下であるとhash比較した。通常renderは反復しない。
証拠は`target/i10-joint-shared-state-20260920/summary.json`、SHA256 `4d399ca12e7f53a07b6fe8ef1d28013f87eb840825184664b3ad0b7391a9d5a6`。

keyの生成元はまだ呼出し側であり、最も強く支持されたretrieved／new共有proposalの実証拠選択と結合は未完。
summary項目の意味を登録した段階であり、stage3の全条件付きlog-potential座標・fitを代替しない。
次は実共有producerと全score、joint contextのcue確定への投影、通常接続を進める。
全候補帰結、実身体転用、残るdefault、統合・資源引渡しを含むI10全体は未完了。
I9保留、I11無効、at-action評価維持。R2／A3は別工程。


## 実証拠からの共有proposal選択とowner照合（2026-09-20）

共有producerが、既知親ごととunknown親について、stay・unknown・最強の支持付きreturn・最強のnew/contrastを生成する。
元のsection-return cacheを使い、episode ID／generation・確定context・queryを保存する。追加DTWや音響scoreの再softmaxは行わない。
新規候補は前回の局所section履歴と既存Headの遷移則から生成し、今回の局所posteriorを再投入しない。

今回登録した候補生成用の工学則は次のとおりである。alphaは現在区間の割当秒数／物理窓秒数とする。
returnのraw scoreはalpha×元context支持質量。既知sectionのnew/contrastはalpha×前回局所条件付き質量×Headの該当exit質量。
section unknown質量からの初期newはalpha×その質量とし、既存articulationの低energy閾値（RMSが0.01×rms_reference以下）ではadmitしない。
最大scoreを選び、同率はgroup generation・role・episode/section identityで決定する。exact tieは安定した先行候補を保つ。
これらは既存の4-role共有遷移核へ渡す候補生成scoreであり、f_sharedへの追加観測scoreでも校正済み確率でもない。

producer自身が単調なkeyを所有する。現在のadmission identityと一致する候補は親keyを使い、遷移核が重複roleを抑制する。
弱い別候補を詰め直さない。ownerで拒否された同じcutを再生成できるが、割り当て済みkeyは再利用しない。
producerエラーではcounterとcutを変更しない。最大31占有proposalを32 slot内に保持し、生成時のheap確保を追加しない。

生成batchは共有親・role・key・transition・区間・group順序を固定する。
ownerは元の音響支持と現在のepisode保持を正規化前に照合し、採用したadmission根拠を共有Stateへ保存する。
stayに残る過去のquery／cutは歴史的根拠であり、freshな復帰支持へ読み替えない。
局所sectionの境界やcontextを共有candidateに合わせて強制しない。

5更新で22共有proposal、独立式によるcontrast選択、静音・欠測・同率・group順序変更と7拒否後の再試行を検査した。
contrastの独立式は既存Headのsoftplus hazardを使い、今回の固定intercept log(4)では積分が0.1×log(5)となる。
実cue／Recall／matcherの正のcontext支持2条件で、計12共有復帰と12退役拒否を検査した。
共有質量を元context支持からの独立式と照合し、query・episode世代・context・観測区間の保持を確認した。
各条件の最初の6支持窓の検査であり、独立したquery数ではない。key overflowと同じcutの再生成も検査した。
Stateは4,600 bytes、snapshot本体は378,520 bytes（heap除外）。全workerの資源受入ではない。

共同関連47テスト、全Rust 1,296成功・失敗0・38 ignored。
fmt、通常Clippy、全target check、release build通過。cargo test exit=0 @ 2026-09-20T12:51:20+09:00。
変更9 sourceはすべて変更していないcfg(test) joint配下と照合したため、通常renderは反復しない。
証拠は`target/i10-joint-shared-producer-20260920/summary.json`、SHA256 `6305ee6ca8456de5713a3eec946084b6be9fe50f7f3b3f493174a5048bfe64cd`。

局所tupleの全producer、joint contextのcue確定投影、全順序付きf_shared/f_g座標と実score、通常共同推論・全帰結接続は未完。
単体fixtureには生成batchなしの制御入力も残る。通常統合では生成batchによる照合を使う。
実身体転用・残るdefault・統合・資源引渡しを含むI10全体は未完了。
I9保留、I11無効、at-action評価維持。R2／A3は別工程。


## 局所section候補の生成とcorrespondence整合（2026-09-20）

元のObservation、前回の局所section、既存Head、確定済みReturnからsection候補を生成する経路を追加した。
復帰先は元context支持質量の最大値、同率ではcontext・episode世代・anchor・correspondence ID順とする。
支持質量を再正規化せず、既存Headのreturn exitに掛ける。同じcontextへの復帰はDevelopmentとし、境界Stayを保つ。
unknownからの新規contextは正の音響割当支持と既存低energy閾値超過を要求する。
欠測時は既知section stayと120秒尺度のunknown leakのみ。新しいcontextや復帰をadmitしない。
IDはまだ呼出元が割り当てるが、親・新context・既知復帰先との衝突と不正な音響時計を拒否する。

組合せ器はphrase×sectionに加え、section×correspondenceの整合を生成段階で検査する。
復帰には同一query・episode世代・元支持を持つfresh correspondenceを要求する。
sectionごとの最良合法correspondenceを先に選び、各phrase/section境界区分の最良合法tupleを予約する。
追加のcomposer探索は最大95 boolean検査。16候補、16 heap pop、86優先度評価の上限を維持する。
95検査に候補構築・証拠照合の全費用は含まれず、最終worker censusで別に数える。

独立Decimal全直積参照を24条件・120,996組へ拡張し、予約・順位・正規化・上限と照合した。
既存12条件は旧generatorから再構成して同一性を確認した。
実matcherの12支持窓では363 tupleを生成し、90不整合cellを排除、36復帰／Development経路をownerへ採用した。
採用後のquery・focus・contextと元correspondenceを照合し、cached支持への置換、欠測、静音、ID衝突も検査した。
これは2条件×最初の6支持窓であり、36独立queryではない。前回section質量とHeadは制御値を使い、
この焦点検査では他の3成分をunknownとした。全5成分の通常推論の証明ではない。

共同関連48テスト、全Rust 1,297成功・失敗0・38 ignored。
fmt、通常Clippy、全target check、release build通過。cargo test exit=0 @ 2026-09-20T13:07:48+09:00。
変更11 sourceはすべて変更していないcfg(test) joint配下。通常renderは反復しない。
証拠は`target/i10-joint-local-sections-20260920/summary.json`、SHA256 `49926c80527903c17e2feef6c058c29336a26803745de0b251067a269393004f`。

全成分producer、全親／共有候補間のID割当とsource結合、phrase ending・joint cue確定投影は未完。
今回のsource生成ではcurrent retrievalの2座標もまだmaskされ、完全なimmutable query inventoryからの投影が必要である。
全条件付きscore、通常共同推論・全帰結、実身体転用・残るdefault、統合・資源引渡しも未完。
I10全体は未完了。I9保留、I11無効、at-action評価維持。R2／A3は別工程。


## query全件のretrieval特徴とphrase音響投影（2026-09-20）

correspondence cacheは、16候補への切捨て前の元query全件から最良scoreと異なるepisodeとの差を保存する。
通常sectionと共同経路で同じ集約則を使い、比較対象のepisode ID／generationも保持する。
元cue支持・受信時刻・0.5秒の支持年齢・deadline・退役を照合し、欠測ではmaskする。
追加DTWやheap確保は行わない。全候補の既存音響scoreの走査費用は最終censusに含める。
局所section sourceへこの2座標を供給し、owner入口で元cacheと照合する。
80候補の上位16件が同一episodeの変形だけでも、外側に残った別episodeとの差を独立式と照合した。
12実matcher支持窓では投影値の差替えを拒否し、採用されたsectionの80／81座標まで確認した。

共同経路の音響cacheは、解釈質量を持たない元raw・energy・accentを物理2秒の範囲で保持する。
既存の短窓集約と六記述値集約を使い、前回cut／現在cutの音響特徴と各phraseのendingを投影する。
ownerはcacheと元Observationの一致を先に検査し、pruning後の保持phraseにのみendingを適用する。
完了済みphraseのendingは元の聞こえた終端を保ち、境界hopの音で更新しない。
30更新・24独立moment照合、欠測・原時計・音響差替え、200 sample／128 accent容量を検査した。
accent容量損失は密度のunknownへ反映する。bufferは構築後に増大せず、全負荷の時間受入はまだ行っていない。
既存owner検査の622経路にも投影し、section unknown下のphrase保持と欠測を確認した。

共同関連51テスト、全Rust 1,300成功・失敗0・38 ignored。
fmt、通常Clippy、全target check、release build通過。cargo test exit=0 @ 2026-09-20T13:23:36+09:00。
通常側の共用集約を変更したためsteady／amplitude／pulseの3実renderも行い、WAV完全一致、
393候補record・434時間観測と既存意味出力の一致を確認した。処理時間・資源counterのみ比較から除外した。
証拠は`target/i10-joint-evidence-projection-20260920/summary.json`、SHA256 `0b0afef90fb7cdfbff5f963986b8345a587f43df0be63de28a57f0c2f3b85551`。

共同経路はまだcfg(test)。全成分producer、残る条件付きphrase特徴、joint cue／context確定投影、
全順序付きscoreと通常共同推論・全帰結、実身体転用・default・統合・資源引渡しは未完。
I10全体は未完了。I9保留、I11無効、at-action維持。R2／A3は別工程。


## 全5成分の局所候補生成と不変source結合（2026-09-20）

shared候補・各group・保持された各局所親とunknown親について、articulation／grouping／phrase／section／correspondenceを生成し、
既存の有界composerからownerへ渡すproducerを追加した。phrase／sectionと新しい局所contextのIDは単調に割り当てる。
同じcutの再計画でも発行済みIDを再利用せず、score失敗・ID overflowではcounterを更新しない。
確定前のbatchは元Observation・cache・modelを不変借用し、別のsourceへ差し替える入口を持たない。
Returnの証明は値として事前確保したsource配列に保存する。1親16 tuple／86優先度評価、
全体65,536 row／16,384 sourceの容量を維持し、group退役後の空inventoryにも対応した。

欠測中のphrase／sectionは、duration hazardの生存確率とunknown漏出の生存確率を掛けてstayを残す。
退出と積分失敗はunknownへ送る。stayだけを再正規化してhazardを消す処理を修正し、32条件の独立式と照合した。
phrase headには前回保持articulation、前回cutの実音短窓、前回選択correspondenceの支持付きcostを供給する。
grouping-admission平均とbootstrap質量の残座標は依然として上流入力であり、joint事後確率で代用していない。

8更新・21,943 tupleでは、各局所partitionを含む共有証拠と保持確率を通常確率の独立列挙と照合した。
欠測ではpotentialを0へmaskし、観測支持のない回復を作らない。続く全group退役も検査し、buffer容量は増大しなかった。
実matcherの12支持窓を、制御した周期accentのgrouping有無で24条件として検査した。
全5成分が既知のtupleは540件生成・360経路保持。groupingありでは帰還候補は競合によりbeam外となり、
grouping欠測条件では126件の合法な帰還tupleを生成した。切捨てを隠さず、全合法帰還の保持を保証したとはしない。

共同関連53テスト、全Rust 1,302成功・失敗0・38 ignored。
fmt、通常Clippy、全target check、release build通過。cargo test exit=0 @ 2026-09-20T13:48:08+09:00。
変更したsource 9件はすべて既存のcfg(test) joint内。通常sourceは前回hashと一致するため、今回はrenderを再実行していない。
証拠は`target/i10-joint-local-producer-20260920/summary.json`、SHA256 `4294b4cd048bb081b78515d671eb59dc829f69b4c551078270ea6241c38f68a4`。

全条件付きscoreは引き続き明示的なcallback入力。検査用scoreを本番の特徴組立・適合完了とはしない。
残るstage1 phrase入力、joint cue／context確定投影、全順序付きscore、通常共同推論・全帰結、
実身体転用・default・統合・資源引渡しは未完。I10全体は未完了。I9保留、I11無効、at-action維持。R2／A3は別工程。


## I10の範囲確定と残作業の再定義（2026-09-20）

2026-09-20に採用した機構選択規則（[technote-ledger §9.3.55](../../design-notes/technote-ledger.ja.md#mechanism-selection-rule)、
[milestones §1.3](milestones.md#13-機構選択規則による本体範囲の縮小2026-09-20同日のレビューを反映)）により、2026-09-18の
「I10完了境界」の問いは、正本 milestones §2 の狭いI10、すなわち「実Voiceの身体・実行結果→私有trace・候補帰結の
数値経路→診断」で確定した。共同推論・全head・全資源受入を含む拡張範囲は採用しない。これは完了しやすさを理由にした
読み替えではなく、和声側でR/Hと慣れだけをDCCに置く分担を時間構造へ適用した結果である。実装状態は不変であり、
変えたのは完了対象と残作業の分類である。

### 研究拡張の保存記録とする節の範囲

- 対象は2026-09-18〜2026-09-20の節のうち、見出しに「共同」「section」「phrase」「correspondence」「retrieval」「grouping inventory」を含むもの。
  具体的には「通常section経路の更新IDと親参照を公開（2026-09-18）」「一つのarticulation状態に条件づけたphrase headを共用（2026-09-18）」
  「共同正規化の0音群境界を修正（2026-09-18）」「sectionの共同遷移adapterと展開時の境界を検査（2026-09-19）」と、
  「共同経路の前回重みと親identityを保持する層（2026-09-20）」から「全5成分の局所候補生成と不変source結合（2026-09-20）」までの各節。
  本節自身は含まない。
- これらはphrase履歴・section復帰・correspondence／matcher proposal・retrieval特徴・groupingを結合する共同posteriorと全帰結投影の
  実装であり、T5・T6と、T4のうち変形をまたぐ対応に属する。研究拡張の保存記録とし、I10の依存には戻さない。
- `src/temporal_cognition/joint/`、`consequence/`、`long_form.rs` はcfg(test)限定のまま凍結する。以後I10として作業しない。
  削除やattic退避は別に判断する。tuple列挙、matcher支持窓、section arenaの予約量などの数値検査結果は、研究拡張を再登録するときの参照として保持する。

### 2026-09-19監査の残依存6件の仕分け

| 残依存（`target/i10-completion-dependencies-20260919/audit.json`） | 仕分け | 満たさない条件と理由 |
|---|---|---|
| 通常観測経路の共同posterior | 研究拡張 | 条件2・3。phrase／section／correspondence文脈に条件づけた推論で、その文脈側が研究拡張 |
| 全候補帰結投影 | 研究拡張 | 条件2・3。三head候補核の入力が共同posterior |
| 通常共有候補分布（phrase二headの再重み付け） | 研究拡張 | 条件2・3。closure／continuationはT5由来 |
| 参加意欲head（groove／desire、54入力・109係数） | 研究拡張 | 条件2・3。係数は人の評定への適合が前提。通常接続済みの評定headは `[temporal_groove]` 明示時だけの既定off診断として保持し、追加実装しない |
| 実身体転用の妥当性 | 本体残作業 | 感覚運動予測の核心。実身体・残存tone・制御・位相を含む投影で順位反転を解消する |
| I10資源引渡し | 本体残作業 | 縮小した実装負荷を棚卸し・測定しR2へ渡す |

本体残作業は上の2件に、§1.2の契約「onsetのみの成立はI10全体の完了ではなく、release、任意身体、私有trace、候補帰結表、
両busへの展開を残す」に由来する「release・任意身体・両busへの候補評価の展開」を加えた3件とする。この3件目は旧6件の
分割でも新たな不足でもなく、2026-09-17の監査表「onset／release指令と私有実音」「候補帰結の数値経路」行の未完部分である。

### 共有候補表の列の仕分け

- arrival、accent密度の列はI10本体の入力とする。
- groove／参加意欲の列と、phrase closure／continuationの列は、それぞれ `[temporal_groove]`・`[temporal_phrase]` を明示した場合だけの
  既定off診断とし、I11の帰結評価には使わない。`src/temporal_cognition/phrase/projection.rs` の「Raw base-head aggregation only」という
  出所表記を維持する。
- section・recall・correspondence由来の値は通常表へ追加しない。

### I11の入力契約（本体、機構は一つ）

I11で比較するのはonset-nowとdelayed onsetの二候補であり、比較項は「予測した自声の持続・release tail」と「外部音の短期予測」の
同一区間での重なりと、到来期待との関係である。共同posteriorは介さない。各入力の供給元は凍結対象に依存しない。
下表は供給元の登録であり、設計途中である。両候補を同じ観測時点から予測する契約、比較量と集計尺度、有界窓の上限、
予測の情報締切と遅延量、自声・外部予測がunknownのときの候補選択規則は、I11の実装前登録で確定する。登録前はI11の機能成立を主張しない。

| 入力 | 供給元 | 候補時点と予測区間 | 欠測時 |
|---|---|---|---|
| 自声の予測energy・持続・release tail | `src/life/self_prediction.rs`（発行時予測）、`src/life/action_candidates/energy.rs` と `src/runtime/body_profiles/policy_default/energy_projection.rs`（候補・defaultの条件付きenergy） | 各候補のonsetから身体envelopeの終端までの有界窓 | 私有実音の欠測は学習なし。候補評価はunknownとし帰結creditを与えない |
| 外部音の短期予測 | `src/core/temporal_expectation.rs` の `OwnSoundHistory::predict_at`（自声除外後の帯域energy） | 候補ごとに同じ区間を評価 | 予測未発行ならその区間の重なりはunknown |
| 到来期待 | `src/temporal_cognition/arrival.rs` の `Engine`（観測accent時計上の次accent予測） | 候補onsetと次accentの時間差 | 非拍節・欠測ではunknownとし、到来項を比較から外す |
| 作者の参加傾向 | `src/life/temporal_participation.rs`（固有周期・参加周期・鳴る／待つ／見送る） | 比較結果を発音時刻へ返す唯一の経路 | 既存の既定動作 |

I11の完了条件は、この二候補の比較が固定方策のoff基準と可聴差を生み、作者が採用すること（A1／A2）である。
「全候補帰結投影」の除外はこの直接比較を妨げない。

### I10完了監査表（2026-09-20、狭い範囲）

| 要件 | 必要な証拠 | 現在 |
|---|---|---|
| onset／release指令と私有実音、任意身体の六記述値、動作条件付き自声予測、私有参加trace | 旧監査表（2026-09-17）の各行 | 通常接続・検査済み。本体への作者採用はA1／A2で判定する。認知機構の妥当性を実証したことは意味しない |
| 実身体転用の妥当性 | 実身体・残存tone・制御・位相を含む候補energy予測が、offline実音比較で順位反転を残さない。失敗した身体の除外で合格にしない | 未完。sine backendは1,520候補対で反転0。非sine等は明示的に旧近似 |
| release・任意身体・両busへの候補評価の展開 | 固定方策の音声不変、候補による学習不変、routing・世代・欠測の検査 | onset候補は接続済み。release分岐は取得と一部接続。任意身体は転用妥当性の解決に依存 |
| I10の資源 | 上記の実装負荷を4／16／64 Voiceで測定しR2へ渡す | 未完。前版の測定は旧負荷に対するもの |

監査記録は `target/i10-scope-confirmation-20260920/audit.json` に保存する。今回の変更は文書と監査記録に限り、
Rustと生成音は不変である。I9保留、I11無効、at-action維持。R2／A3は別工程。次の一単位は実身体転用の妥当性である。
