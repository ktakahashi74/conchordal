# F4b封印sourceとF4c接続案の独立レビュー

確認時刻: 2026-09-27 00:15 JST。対象は
`target/body-fitness-f4b-capsule-20260927/`の封印source（tar SHA-256
`589020da6d085bfe0eb15983a0e12cc81997f62538f7fcfd8ae644dbdb46181b`、
manifest SHA-256
`883add0aa049e40f86634f7f0db985c05073fca0a1c3741885fd6f3b2df247af`）。
主な対象ファイルのSHA-256は`respawn.rs`が`93198809b0b01124dfeae2e60774d371185a31df4cc777f401293c0f60f36182`、
`respawn/f4b_offline.rs`が`09a75d90abc7d8d295c4781bb426a0ae1f674dbc32de9cf072c001a7ef59417b`。
登録は事前無効取得後の改訂であり、完全な初回事前登録ではない。レビューは読取のみで、新たなrender・testは実施していない。

固定Random一機会の主要配線は整合する。出生時の16候補はCommunityの既存`spawn_seed`と
`random_respawn_frequency`を同順で再演し、`choose_candidate_by_scene_score`は各slotの
身体score正部分を`WeightedIndex`へ渡す。重複周波数もslotを保ち、同一周波数の身体評価・
identity一致を検査する。選択後のlevelだけを別に使って閾値を判定し、拒否では
`spawn_counter`のみ進む。成功時のruntime ID、member index、generation 0、parent None、
子BodySnapshot、周波数bit、RNG fingerprintは実出生経路で照合される。共有環境は旧Toneの
明示Off後の減衰PCMを含み、48 kHzの実AnalysisStreamから`recompute_consonance`を行う。
候補の主観強度密度は同じ解析設定で72 hop測り、共有環境の有効CをERB幅で積分してから
levelへ写す。点Cへの毒入れ対照でRandom身体選択を旧点値から分離できている。

残る表現上の境界は**代表recipeの全要素を実子から独立に照合していない**点。
`validate_candidates`と`verify_child`は期待`entry.recipe`をcloneし、子由来のbody、
frequency、modulatorだけを上書きしてidentityを再計算する。hold、ADSR、
`smoothing_tau_sec`、sample rateは期待値のままなので、その一致は検査されない。
今回の手書き代表recipeはhold 48000 sample、ADSR `(0.005, 0, 1, 0.5)`、
smoothing 0。一方、通常の`Voice`がToneSpecを発行する場合、smoothingは0.02、
ADSRはVoiceのbody envelopeに由来する。このfixtureはデフォルトのSine envelopeを
残してbody methodだけHarmonicに変更している。したがって「実子のTone recipe全体が
採点時と同じ」とは言えない。登録した固定の代表proxyを評価する試験としては成立するが、
結果の表現は「実子由来のbody・基音・modulatorが宣言代表recipeと一致」までに限定する。
F4cで全recipe一致を要求するなら、期待recipeのcloneではなく、子から独立に生成した
代表recipeと全要素を比較する必要がある。

F4c草案の現版（確認時SHA-256
`c9958da772dc0768c7a64e1cbd1c93ea9ec66b0a77b1d9876c7e9b8d7b72d68b`）は、
親poolを生存Entrain二体に固定し、更新後energyを正・相異とする条件、Hereditaryの
`max_by`が読む**全候補level**への身体入力を明記済み。これらは必要。
現F4b hookはRandomのscore抽選と選択後のlevel閾値に限られ、Hereditaryの
`max_by`はまだ点levelを直接読む。次工程ではこの比較入口だけを接続し、
既存の親energy比例抽選、同値規則、候補生成、乱数順序を重複実装しない。
親の音色遺伝や長期選択を、この一機会の試験結果として扱わない。
