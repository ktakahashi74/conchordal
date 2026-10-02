# F4 全消費者比較へ進むための設計草案

2026-09-27。第十八版の作業中ソースを読んだ段階の設計であり、数値条件の事前登録、実装、取得結果ではない。第十八版の反復respawnが通っても、F4全体の終了条件は満たさない。目的は `body-aware-fitness-plan.md` §5の「固定した異種身体の生存差」と「移動のみ／生存のみ／全接続」の比較である。respawnの回数を増やす前に、同じ身体評価がどの消費者へ入り、どこで欠けるかを分離する。

同日の[実装レビュー](body-fitness-all-consumers-implementation-review-20260927.md)で、offline移動は`receive_offline()`による決定的な待ち合わせを使うと確認した。以下のReady待ちを実時間の計算遅延による欠測と一括しない。また、旧flagのOFFは共有地形と旧LOOへ戻るため、身体平均だけの除去比較には使えない。第一単位は全条件で同じsource-removed環境、prepared時計、代謝receiptを用いる出生なし二身体の四条件比較とする。子の出生前評価は自己PCMのない共有環境を使い、親の自己除去環境へ置換しない。これらの具体化は実装レビューを優先する。

## 現在の境界

- 初期 `Action::Spawn` は一つの `VoiceSpec` を全memberへ渡し、`RuntimePopulationState` は一つの `template` を持つ（`src/life/community/actions.rs` の `on_spawn_action`、`ensure_population_state`; `src/life/community.rs` の `RuntimePopulationState`）。複数のSpawnで同じPopulationを指定するとtemplateが最後のspecへ更新され、各Spawn内の `member_idx` は0から振り直される。この既存動作を異種身体fixtureへ流用すると、子の身体とmember identityが暗黙に変わる。異なるPopulationへ異なるtemplateを置けば共有音場での生存比較はできるが、親poolはPopulation内なので異種身体間の親選択を示せない。
- 親付きrespawnは更新後の生存親energyで抽選し、子の評価には確定したID・metadataとPopulationのtemplateから作る代表身体を使う（`src/life/community/respawn.rs`、`src/life/community/respawn/offline_respawn.rs`、`src/life/community/child_body.rs`）。現行の子身体は親から遺伝しない。出生採点は子の実身体に照合されるが、同じPopulation内の親が異種身体である比較は未実施。
- `body_fitness_action` はsourceごとの非同期Readyを待つ。hop前にproposalを保留し、環境またはReadyが欠ければそのhopは保留のまま、適格Readyを採点・installした場合だけ解除する（`src/runtime/body_fitness_action.rs` の `before` / `after`）。要求はrender後に送る（`src/runtime/mod.rs` の `process_hop`）。身体別の準備待ち頻度が違えば、移動機会自体が違う。消費数だけでは移動への等量介入を証明できない。
- offline代謝は各Voiceの `SourceIdentity`、代表Recipe、epoch、space、habituation、support時刻を検査して同期receiptを作る（`src/life/offline_metabolism.rs` と `src/runtime/mod.rs` の代謝install）。親付きrespawnの子はcleanup後に生まれるため出生hopの代謝receiptがなく、次hopから `SourceRemoved`。初回 `InitialShared` と別経路の初回出生 `BirthShared` は異なる由来。休符中の生命更新にも代表身体のscore/levelを使うが、実PCMの有無とreceiptの有無を混同しない。
- 現flagは独立した消費者マスクではない。`body_fitness_respawn_offline` は代謝ONかつbirth/action/observation OFFを要求する（`src/config.rs` の `validate`、`src/runtime/mod.rs` の `wire_runtime`）。別の出生＋代謝経路もaction/observationとの同時ONを拒否する。したがって既存flagを単に全ONにするだけでは比較を走らせられない。
- source集合は生存Voiceだけでなく保持中の死者と新生児を数え、現在の自己除去解析は最大4source。`src/runtime/mod.rs` のrespawn遷移検査はID・系譜generation・birth_sampleを照合する。死亡hop、子の全ゼロ自己PCM、翌hopの新slot/receiptを同じsourceと機会番号で結ぶ必要がある。第十八版の既知の同一hop重なり拒否はcleanup後のrecord処理にあり、親RNGやcounterの更新前拒否を一般保証しない。この境界を直す前に「任意の逐次生態」と呼ばない。

## 最小の比較経路

まず四source上限内で、出生のない固定異種身体の短い生存比較を独立に成立させる。共有環境、同じ時刻・基音・制御・放射量の下で、明示的な二種の固定身体を各Voiceに割り当てる。これは音響条件の選定ではなく、fixtureが宣言した身体を実Voiceと代表Recipeが保持するための経路である。別Populationのtemplateを使う暫定比較は「生存差のみ」と明記し、異種親poolの証拠へ転用しない。

次に一つのPopulationで最大三founder＋出生子一声とする。初期Spawnのmemberごとに固定body specを指定し、member indexとsource identityを一意に保つ最小の入力を追加する。子の固定body specは親抽選前に明示して確定し、親の身体を写さない。既存のSpawn周波数選択、親重み付きRNG、Peak/Density・占有、全bin／局所／最終Hz採点を再利用する。Population templateを最後の初期memberに暗黙依存させない。通常Rhai表面を増やすか、研究用の通常offline Scenario入口で固定member specを渡すかは未決。どちらでも通常rendererの実Voice生成・通常cleanupを通すことが条件であり、`cfg(test)` hookのみでは足りない。

比較用にはaction、代謝、生存・親energy、出生候補の各消費を独立にON/OFF指定できる小さいofflineマスクを設ける。OFFは候補・時刻・source除去・RNG規則を消す意味ではなく、当該消費者だけ点評価へ戻す意味に固定する。三つの主条件は「移動のみ＝action身体ON、代謝/出生候補は点」「生存のみ＝action点、代謝身体ON、出生候補は点」「全接続＝三者とも身体ON」。親energyは代謝の結果なので、生存のみ条件の親重みも変わり得る。親選択の直接効果と出生地点評価の直接効果を識別するには、親重みを点条件で固定する追加対照、または親pool/抽選確率を条件付き解析する必要がある。点・点・点の基準条件も保存する。現在の複数boolean flagをこの意味に読み替えず、同時flag拒否の解除とマスク配線を一つの登録で明示する。

全条件で初期source、固定身体割当、Seed、解析空間、habituation、48k/512、route、Finish規則を共有する。ただし生存・出生が変われば後続のsource集合とRNG呼出し列は分岐する。同一SeedのWAV一致を異条件間の必要条件にせず、条件内再現性と死亡・出生・親poolの実系列を記録する。出生後も四sourceを超える機会は事前に拒否する。第十八版の二機会sceneをそのまま比較条件へ拡張する必要はない。

## 欠測・時計・失敗条件

各判断についてactionのReadyあり／遅延／拒否／保留、候補score使用数、実target変化を記録する。代謝receiptと出生判断記録を `source{id,generation,birth_sample}`、body generation、Recipe/route、epoch/space/habituation版、`support_end_sample`、`decision_at_sample` で照合する。出生機会は実親poolの更新後energy、spawn sequence、RNG probe、子の予定/実identityとbodyを保存する。出生hopの子を旧batchの既存source扱いせず、次hopには子自身の自己除去slotを使う。音色ごとのReady欠測が生存差に見える危険を避けるため、必要な有効判断率、欠測hopのaction保留・代謝更新・出生保留の規則を取得前に固定する。現行の厳格な同期代謝が欠測をpanicとする範囲と、将来の待機規則を区別する。

先に同一hopの「前児next確認」と「新出生」をcleanup前にpreflightするか、両遷移を順序どおり処理できるようにする。第五source、未解決の前児receipt、stale Ready、source世代/birth sample違い、環境support/epoch/space/habituation違い、子Recipe違い、親pool stale energyをRNG・ID割当前に拒否する境界を検査する。初期の独立比較では一般の同ID再利用、長期生態、音色遺伝、全音色への計算費用・有効率保証を主張しない。

次の登録で決める事項は、固定二身体の宣言値と検査許容、同一Populationへのmember指定入口、子の非遺伝body指定、必要なaction有効率、欠測時の生存/出生時計、親重みの追加対照、観測量と期間である。取得値から都合のよい身体・Seed・死亡時刻を選ばない。F4の判断は生存時間、energy系列、親選択、出生候補と実Hz、有効評価率を条件別に比較してから行う。資源計測、作者採用、既定変更、F5の操作面、F6の音色遺伝は別の判断とする。

## 2026-09-27 19時台の確定版から見た生存比較の残条件

上の初期設計で未実装だった異種founder入口、固定子身体、三軸切替、同hop遷移は、[三消費者v2](../../.worktrees/body-fitness-three-consumer/docs/design-notes/body-fitness-three-consumer-results-20260927.md)で限定的に成立した。以下はその422ファイル固定版を読んだ次段階の設計判断であり、新しい数値取得の登録や結果ではない。

22hopの実験ではbackground death rate 3/秒が出生機会を作った。`Community::apply_background_turnover`は各生存Voiceに`rate * dt`の確率で除去を指示し、このhazard自体はenergyを使わない。energy差が親抽選重みに届いたことは確認したが、その背景死亡を身体評価による飢餓死の差とは解釈しない。現在の初期Sustain設定はendurance 2秒・recovery 10秒・dissonance penalty 0・attack cost/recharge 0であり、連続時間の近似では基礎消費0.5/秒と回復0〜0.1/秒となる。0.2347秒の観測だけでenergy枯渇による生存差を測ったとは言えない。実装は基礎消費後と回復後に個別に枯渇処理をするため、正確な到達時刻はsubstep順序で扱う。energyの最初の0到達、retrigger停止、ArticulationのIdle、実PCM tail終了、source退役は別の時刻として記録する。

単にFinishを延ばす前に、次の実装境界を解消する必要がある。現`validate_offline_respawn_metabolism_scenario`はbackground death rate > 0を必須とし、三消費者validatorもこの検査を再利用する。したがって背景死亡なしの生存実験は、現在の入口では実行できない。また`OfflineRespawn::prepare`は生存親2声・一機会一死亡を要求する。複数個体のenergy枯渇が同じhopへ重なると、長期生態の一般的な受理範囲を外れる。source上限4は死者tailを含むため、単純に生存数3だけを数えて容量を保証しない。これらを無視するためにseedや身体を選び直さず、有限の生存比較と出生を含む比較の契約を別に固定する。

生存差の次の候補は、背景死亡とrespawnを止め、宣言した異種founder全員を期限まで追跡する有限コホート比較である。遺伝がなくても、各founderのenergy枯渇・音響的終了・生存時間と有効評価率を比較できる。三消費者全接続の比較を置き換えるものではなく、その生存経路の効果を識別する追加対照とする。身体と初期f0の割当を事前に交換し、通常ampと固定放射RMSの条件を分離する。周波数移動による環境変化と代謝scoreの直接効果も、点／身体の対応条件と最初の分岐で区別する。現登録に結果を継ぎ足さず、場面・期間・seed集合・打切り規則・有限コホートの実行入口を別登録する。

その後の出生を伴う固定身体比較では、Populationの一つの固定子templateがfounderの異種性を失わせる点も扱う必要がある。現在の固定Sine子は初回の身体照合に適するが、長期に異種個体の流入を保つ実験ではない。親から身体を写すとF6の遺伝介入になるため、F4の追加流入を設計する場合は親選択と独立した固定割当・流入則を事前登録する。親重みの変更と実際に選ばれた親の変更、出生場所の変更、founderの生存差を別々の観測量として残す。

直近の実行単位は既定どおり、固定22hopの実消費記録と独立再計算である。その結果を閉じてから、背景死亡0・有限コホート入口、同hop複数枯渇とtail容量、身体／f0と放射量の対照を具体化する。実時間性能・作者採用・F5/F6の終了条件はこの設計で満たしたとは扱わない。
