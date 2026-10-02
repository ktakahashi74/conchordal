# 複数 respawn と固定異種身体の比較に向けた境界整理

状態：2026-09-27 の第十七版を読んだ次工程の設計 draft。[第十七版fの固定取得](body-fitness-offline-respawn-results-20260927.md)では、同一 Population の固定 Harmonic Entrain founder 3声から、通常 offline renderer で親あり PeakBiased respawn を実際に一回行った。OFF は身体代謝・身体出生をともに無効、ON は両方を有効にした結合比較であり、代謝だけ、出生だけの効果は分離していない。第十七版fの全 suite は1322成功・0失敗・48 ignore。非PeakBiased診断修正、通常cleanupの回帰試験、通常render再取得を含む。この文書は次工程の数値登録ではない。今回、実装・追加試験・render・性能取得は行っていない。

拡張前に[検査範囲表](body-fitness-offline-respawn-focused-review-20260927.md)の未確認項目を照合する。peak抽出の独立再実装とruntimeへ直接注入していないfaultは、現版の全test成功でも独立検証済みにはならない。

## 現行の一機会が固定しているもの

`runtime/mod.rs::validate_offline_respawn_metabolism_scenario` は、時刻0の同一 Population・Entrain founder 3声、一つの Linear 配置、一つの PeakBiased policy、容量3、単一 Finish だけを受理する。Harmonic 身体は登録sceneの固定条件であり、validator が全シナリオへ課す条件ではない。途中 Spawn、Release、control、解析変更と action／observation／初回 Field 出生 flag の併用は拒否する。`WorkerState` の `respawn_opportunity_seen` は二度目の記録を拒否し、初回 founder ID だけを死亡元とみなす。登録sceneでは子ID4、member3、系譜世代1を実出生後に照合した。`OfflineRespawn::attempted` と一件だけの `records` も同じ制限を持つ。これは一般の respawn 生存過程ではない。

`Community::respawn_on_new_deaths` は各 hop の lifecycle と phonation 入力後、生存 Voice の実 energy から親 pool を作る。現在の opt-in 評価器は、単独新規死亡・親2声・予測上の第5 source がない条件で、子の予定 ID／member／系譜と身体候補を確定する。最低 level 拒否でも `spawn_counter` は進み、ID／member は消費しない。複数死亡は現在、`death_observed` を変更する前に失敗させる。一般化時も「一機会の識別、親選択、拒否後の counter 状態」を単位として保持する必要がある。

自己除去の worker と observer は最大4 source、48 kHz・512 sample/hop、支持終端から判断まで最大4,800 sampleを受理する。`SourceIdentity` は ID、系譜 generation、`birth_sample` の組である。新 source の `birth_sample` は、その source が初めて観測に現れる frame の開始 sample と一致しなければならない。死亡 Voice が `should_retain()` により観測集合へ残る間は、その Voice も4枠を使う。Tone の残響は、Voice が除去された後も共有PCMへ残り得るが、退役済み source の自己PCMとして差し引かない。現在の runtime ID allocator は新 ID を選ぶため、同 ID・別世代再利用は通常経路で未検証である。

## 次の最小単位：同種身体で二機会

次に正当化できる最小 gate は、同一 Population・固定 Harmonic 身体・現行の代謝と親あり出生だけを維持し、時間の異なる単独死亡二回を通すこと。第一子の翌 hop の自己除去 receipt を確認した後に、二度目の機会を置く。第一子が二度目の親 pool に入るか、退役 founder がまだ source 枠に残るかを取得前に固定する。二度目の出生を許す場合は、出生前の観測 source 数を3以下に抑え、子追加後も4以下とする。二回とも死亡前の全 source、死亡後の保持集合、子追加後の集合を実 Voice の `SourceIdentity` で照合する。第5 source が見込まれる場合は、親 RNG や ID 割当前に明示的に拒否する。容量を暗黙に超えて source を削る運用にはしない。この gate に異種身体、移動判断、実時間性能を混ぜない。

`OfflineRespawn` の一回限りの `attempted` を、機会ごとの状態へ分ける必要がある。機会キーは少なくとも hop、死亡 Voice の完全な identity、spawn sequence を含め、前機会の候補表・予定子・記録を次機会へ持ち越さない。拒否機会も counter の変化を報告する。`WorkerState` の単一 `respawn_child`、`respawn_dead_id`、`respawn_next_frame`、`respawn_opportunity_seen` と founder 固定照合は、進行中の出生／翌 hop receipt を機会別に追跡する有界状態へ改める。報告も機会 ID で出生、翌 hop、拒否を結ぶ。全bin・局所候補を無制限に蓄積せず、機会ごとに排出する。

出生 hop の子は cleanup 後に追加され、同 hop の lifecycle・phonation 代謝には参加しない。描画前に子の無音512 sample自己PCM slotを準備し、描画後の observer が出生前共有履歴を clone して新 source の初回観測を進める。子の最初の `SourceRemoved` receipt は翌 hop 以降であり、出生 hop の receipt を補作しない。子が翌 hop 前に退役した場合は「初回 receipt なし」を明示する。`OfflineMetabolism` は Voice ごとに source identity を固定し、身体世代を系譜世代から独立に数える。二代目以降でもこれを混同しない。

次の登録では、音声・身体 score・energyを見ずに seed、死亡率、Finish、許容する二つの単独死亡と期待 source 数を先に固定する。RNG-only で死亡時刻を予測しても、実 lifecycle による生存は条件付きと明記する。二回目の死亡、親 pool、ID／member、候補身体、実子、翌 hop receipt を同一試行内で照合する。第5 source、同時死亡、親不足、支持欠測・期限切れ、epoch／space／habituation 変更、古い候補再利用、birth_sample 改竄、無音自己PCM欠落、二件の記録混線は負例。条件不達を取得後の seed や閾値調整で成功へ置き換えない。まだ数値登録は作っていない。

## 固定異種身体と F4 の除去比較

現在の Population は共通 template を子に使う。親の身体を子へ継承していない。したがって現行3 founder の親 energy 差と子の出生先から、異種身体間の生存選択や音色遺伝を結論できない。次の生存比較は、Harmonic と Modal など固定した異種身体を同じ共有環境に置き、各 Voice の身体・音量・初期 energy・配置を取得前に固定する。別 Population の固定 template を使う場合、親選択は Population 内に閉じるので、異種親どうしの出生競争とは区別する。同一 pool で比較する場合は、個体ごとの固定身体指定と子の身体を親から独立に決める規則を先に定義する。どちらも genotype 継承・変異と呼ばない。

移動のみ／生存のみ／全接続の比較には、行動用 prepared 判断、代謝用 receipt、出生候補の身体評価を個別に有効化する配線と、同時 flag の検証境界が必要である。現行 `body_fitness_respawn_offline` は代謝との同時 ON を必須とし、action・observation・初回出生との併用を拒否する。単に gate を外すだけでは、行動の非同期準備時刻と代謝／respawn の環境支持・source identity が一致する保証はない。各判断の環境 origin、支持終端、epoch、空間、habituation、Recipe 世代、欠測期間の保留・energy 更新規則を揃え、異なる時刻の値を同一 hop の「全接続」として混ぜない。乱数消費も処置で分岐するため、同じ seed だけで同一軌道の因果対照とは呼ばず、事前登録した paired 条件と複数 seed の分布で評価する。

生存時間、energy 推移、親選択、出生先、身体 score と点 score の差、準備値の有効評価率、欠測・拒否率を別々に記録する。固定異種身体の比較を終えても、長期生態、音色 genotype の継承・変異、同 ID 再利用、通常の action／observation 同時接続、実 device、F4/F5/F6 全体の完了にはならない。第十七版は隔離 worktree の通常 offline 経路を一機会だけ検証した段階であり、main 採用・既定変更は未実施。実時間のhop予算・欠測率・資源上限も第十七版として未取得で、旧版の計時結果を転用しない。
