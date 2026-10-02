# 通常 offline の親あり respawn と身体評価：次段階の設計案

状態：2026-09-27 時点の第十六版dソースを読んだ draft。初回Field出生と代謝の限定結合は1311テストを通過したが、親あり respawn は未接続であり合格扱いしない。この文書は取得前登録ではない。実装、試験実行、render、性能取得は行っていない。

## 現在の境界

`body-aware-fitness-plan.md` の F4 は、移動だけでなく代謝、生存、親選択、出生先の候補評価を同じ身体 score/level でつなぐ課題を残す。現行の `OfflineMetabolism` は Voice の代表身体を72 hop評価し、実際の lifecycle と onset に入力する。第十六版の初回 Field 出生との結合は、出生前の完全な source 集合を検査し、新生児だけ `BirthShared`、翌 hop から `SourceRemoved` へ移す限定実装である。通常の親あり respawn は対象外で、代謝単独のシナリオ検証も `SetRespawnPolicy` を拒否する。

親あり `PeakBiased` の既存 `respawn_on_new_deaths` は、`cleanup_dead` 時点で生存 Voice を列挙し、Entrain の**その時点の実 energy**を親重みにする。重み付き親抽選を一回行ってから、候補 bin、親周波数による重み、局所探索、最低 level の順で選ぶ。現行の周波数評価は共有地形の点 score/level。`f4b_offline.rs` と `f4c_offline.rs` の `#[cfg(test)]` hook は別に生成した身体評価値を差し込んで、親 pool、RNG、全 bin と局所候補、最終実子を照合する。これは試験専用の一機会であり、通常 renderer が身体評価値を選択に使った証拠ではない。

実行順は `wait_for_analysis` → habituation → `advance_population` 内の環境 install → phonation batch収集（onset時の代謝入力を含む） → `Community::advance_with_listener_pressure` によるlifecycleの代謝・energy 更新 → `Community::cleanup_dead` 内の死亡検出・親選択・respawn → 音声描画 → `BodyFitnessObserver::observe`。したがって、子は出生 hop の通常 lifecycle 更新と phonation batch収集には参加しない。初回 Field 子のように dispatch 直後に `BirthShared` を install して同 hop で代謝した、と見なしてはならない。

## 次の最小接続

新しい明示的な offline 専用 flag を代謝との併用時だけ許し、既定 OFF と instrument 拒否を維持する。対象は決定的な `conchordal-render`、固定解析構成、時刻0の同一 Population の Entrain founder 3声、親あり `PeakBiased` の最初の respawn 一機会に限る。通常の初回 Field 出生 flag、action、observation との併用、Random/Hereditary、複数 respawn、途中 Spawn/Release/解析設定変更は別扱いとする。シナリオ gate はこの範囲を実際の `Action::SetRespawnPolicy` と終了時刻に合わせて新設し、代謝単独 gate を一般的に緩めない。

`WorkerState` が当該 hop の主解析の支持終端、epoch、Log2Space、適用済み habituation、代表評価用 `AnalysisStream` を持ち、`cleanup_dead` の一機会だけへ可変参照で渡す。親の lifecycle を先に実行した後、生存親 pool 全件の `(id, generation, freq_hz, energy)` を記録する。子候補の採点前に既存の重み付き親抽選を一回行い、選択親の世代から子 metadata を決める。現行規則どおり `spawn_counter` は拒否時も進み、子 ID と `member_idx` は成功時だけ消費する。候補 Recipe に必要な予定子 ID は次の未使用 ID を参照し、先に割り当てない。実出生後、予定 ID、親 lineage、世代、member、身体、最終 Hz、Recipe identity を照合する。

`OfflineBirth::evaluate_child` の固定世代0・親なし構築をそのまま転用しない。代表身体の生成と72 hop評価の共通部分だけを再利用し、選択済み親に対応する `VoiceMetadata` と実テンプレートで、許容範囲の**全 bin**を評価してから既存の peak 抽出へ渡す。局所探索で新しく訪れる周波数と最終 Hz も同じ子 metadata・Recipe で評価する。Consonance の score/level と密度 mass の役割を混ぜず、`PeakBiased` の親周波数重み、RNG消費順、fallback、最低 level 判定を保存する。候補生成が拒否・欠測なら点評価や古い表へ戻さず、その機会を明示的に拒否する。出生報告には評価した全 bin、局所候補、親 pool と抽選、支持・決定時刻、epoch、space、実子照合を含める。

`cleanup_dead` は phonation batch収集の後だが描画の前に子を追加する。子のその hop の自己PCMは発声前なので512サンプルのゼロでよい。現行 `prepare_self_sound` は cleanup より前に呼ばれるため、respawn 後、描画前に**新しい Voice の self slot を追加**する接続が必要。既存 slotの履歴を二重に進めず、子の slot と0 PCMを用意する。observerは描画後の実 Voice 集合を完全一致で受理する。新 slot は当該フレーム処理前の共有 `AnalysisStream` を clone し、そのフレームで `mixed - own` を処理する。子は出生 hop に代謝 receipt を持たず、次 hop の開始で前 hop の完全 source 集合から `SourceRemoved` を初回 install して lifecycle に参加する。出生時の候補 fitness と次 hop の代謝 fitness は別の入力・時刻として報告する。

死亡 Voice がまだ `should_retain()` の対象なら observer の source 集合に残り、子を含め最大4声を許す。削除済みでも renderer の音の tail が混合PCMに残り得る。この tail は共有環境に残す一方、存在しない Voice の自己PCMとして差し引かない。親抽選には `is_alive()` の Voice だけを入れる。現行 respawn の `allocate_runtime_id()` は新 ID を割り当てるため、この取得で同じ ID の世代再利用は起こらない。解析器の局所試験は、退役後の同 ID・新世代と tail を独立履歴で検査しているが、通常 respawn の同 ID 再利用を実証したとは呼ばない。活動中の同 ID 衝突は拒否し、将来 allocator 方針を変えるときは renderer の ID 単位 self slot と旧 tone tail を別途検査する。

## 一機会の登録・検証案

取得前に seed、48 kHz/512、Harmonic Entrain の身体と非ゼロ代謝係数、3 founder の初期周波数、`PeakBiased` の範囲・局所探索・最低 level、背景死亡率、最初の単独死亡 frame、Finish を固定する。別の環境 Voice は置かない。3 founder と、退役 Voice が一時残る場合の子で observer 上限4 source に収める。死亡機会は `background_turnover_seed` と各 substep の抽選を先に計算し、親2声が生存する最初の単独死亡 frame を選ぶ。固定不能ならその条件は未登録とし、取得後に seed・率・閾値を動かして成功例に作り替えない。Finish は子の次 hop の代謝 receipt と observer batch まで取得できる時刻に置く。

通常 render の本取得は flag OFF/ON を各2回。ON の同種 JSONL、WAV、実出生、親 pool と親抽選、全 bin・局所候補、出生後の初回代謝 receipt が2回一致するかを確認する。親 energy は出生 hop の実代謝更新後の値で、更新前の値と区別する。独立対照は親 pool 全件の energy とRNGから親を再抽選し、別の代表 Tone/Analysis による少なくとも一候補・最終 Hz の身体 score/level、さらに全 bin と局所候補から選ばれる結果を照合する。共有点 score と身体 score の差を一候補以上で確認する。親の抽選結果や最終 Hz が OFF と異なることは必須にせず、入力重み・候補評価の差と実選択を分けて記録する。子の出生 hop receipt を捏造せず、次 hop の `SourceRemoved` receipt と `birth_sample` を確認する。高い最低 level で拒否する対照は、`spawn_counter` だけが進み、ID・member・Voiceが増えないことを確認する。

必須負例は、親不在・同時複数死亡・5 source目・想定外の追加/削除/世代・活動中 ID 衝突、更新前 energy の流用、親 poolまたは選択親の差替え、子テンプレート/予定 ID/member の変化、候補表と実子の不一致、worker欠測、支持の未来/期限切れ、epoch/space/habituation の不一致、自己PCM slot欠落・511サンプル、出生翌 hop の子 source 欠落とする。拒否は旧点経路への暗黙 fallback を許さない。`#[cfg(test)]` の F4c/F4e は親選択と身体候補計算の既存参照として残し、通常 render の取得とは結果欄を分ける。

この一機会は親の実 energy から一回の出生先選択までの接続だけを示す。長期生態、音色 genotype の継承・変異、同 ID 再利用の通常運転、非同期実時間の有効評価率と費用、作者採用は未完了のままとする。
