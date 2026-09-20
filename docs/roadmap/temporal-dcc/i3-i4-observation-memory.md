# I3・I4: 実観測から音群・記憶照合への接続

対象はユーザーの「I4まで完了させて」に基づくI3・I4。旧M0全体、R/A、I5以降の完了報告ではない。
作業ツリーはHEAD `559d029`以降の未commit変更を含む。以前の905件合格を現版の合格として転用しない。

## 動作と入力契約

I3は既存の音響frontendを、`temporal_mode("observe")`で有効になる二busの独立observerへ接続した。
実音のNSGT scanとmono energyから、ridge、音群配分・成立・退役、音群別raw descriptorとaccentを
同じ観測snapshotへ渡す。UIは保持音群数、生特徴数、accent数と失敗状態を表示する。
energyとrawの所有者は更新前のhandleであり、そのhopで成立した新しい音群へ過去の支持を付け替えない。
原音欠落のepoch変更と配送欠落を区別し、欠測を既知無音へ置換しない。

I4は`recall::Recall`で既存のSpan、prefix、Scheduler、ordered matcher、Controllerを接続する。
入力はI3の音群raw descriptor。残差枠は検索対象にせず、保持された最大7音群を世代付きで追跡する。
各音群の最初の取得hopから明示した`span_hops`に達する区間を保存する。これは診断用の区間であり、
phrase／gesture／sectionの推定境界ではない。欠落を跨ぐ場合は次の配送で終端が確定し、欠落区間を保持する。
退役した音群の未確定区間は破棄し、すでに保存した区間はbus内の共有記憶として保持する。

Spanの登録済み二hop cadence・最大128 knot圧縮を使用する。元NSGT支持、取得量、availability、
尺度、再構成誤差を保持する。queryの評価支持より後へ延びる保存区間を候補から外す。
queryは保存前に発行し、現在区間への自己照合を防ぐ。元NSGT窓同士が重なる可能性は残り、
支持区間の非重複だけで独立した聴取証拠とは扱わない。

Schedulerは登録済み50／100／200 msの取得sample cadence、各音群のpending置換、単一active queryを使う。
複数音群の受付順と処理順は一致しないため、Controllerへ渡すquery IDはdispatch ticketの単調増加IDとする。
元prefixの識別子・支持はScheduler側で維持する。照合計算は専用observer内で実行し、結果を次の観測cut
またはEOFで受け取る。その時点で期限、query所有音群、保存episodeの生存を検査してから診断へ公開する。
期限は取得sample clock上の契約であり、全workerの壁時計遅延・実時間鮮度の合格ではない。
EOFには最後の受信支持とは別に入力終端を渡し、未配送区間も期限判定へ含める。未配送の新epochがあれば
旧記憶の診断を破棄する。期限切れの表示済み結果も消去し、古い支持をEOF時刻へ付け替えない。

保存上限は明示した1–256区間。満杯時は最古の区間を退役させる。退役先を参照するpending結果は拒否し、
表示済みの最良結果も無効化する。epoch変更では記憶・scheduler・controllerを再構築する。
診断にはqueryと最良保存区間の元支持・時刻、候補数、DP cell数、cost、変換、曖昧性を渡す。
空の初回検索は「支持された照合なし」とし、確率や知覚的再認の成功へ読み替えない。
`relations_implemented=false`は採用済みの時間関係モデルがない意味で維持し、`action_enabled=false`も維持する。

## 再現用の明示設定

下記は接続検査用の未校正設定であり、既定設定や採用済みfitではない。省略時は対応する診断を無効にする。
`temporal_acoustic`は`temporal_ridge`を、`temporal_memory`は`temporal_acoustic`を要求する。
非有限値・不正な期間・未対応cadence・未知キーは起動前に拒否する。

```toml
[temporal_ridge]
means = [0.0, 0.0, 0.0]
deviations = [0.05, 4.0, 1.0]

[temporal_acoustic]
group_means = [0.0, 0.0, 0.0]
group_deviations = [0.05, 4.0, 1.0]
accent_means = [0.0, 0.0]
accent_deviations = [1.0, 1.0]
group_retirement_sec = 2.0
inactive_energy_max = 1e-8
correlation_window_sec = 0.25
min_pairs = 8
min_coverage = 0.9
persistence_hops = 3

[temporal_memory]
scales = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
span_hops = 8
episodes = 16
query_cadence_ms = 50
deadline_ms = 200
```

## 検証

- `cargo test --test render_binary temporal_observation_preserves_audio_and_finishes_both_buses -- --nocapture`
  は、off、観測のみ、ridge、音群、記憶の五条件×report有無の計10レンダーでWAV byte一致を検査する。
  実音から複数音群の保存・後続照合まで到達し、無音habitatに音群記憶を作らないこと、元支持、EOF、
  energy保存、handle帰属を確認した。
- `temporal_cognition::observation::tests::acoustic_diagnostics_keep_handles_and_missing_support_across_gaps`
  は、同じobserver入力で原音epoch変更と配送欠落を区別し、欠測・energy・旧handle帰属を検査する。
- `temporal_cognition::recall::tests`は同じ本番Recall経路へ制御したrawを渡し、初出と後続再出、
  順序反転によるcost増加、元NSGT支持、期限超過、音群退役、容量によるepisode退役、欠落と再開を検査する。
  rawを制御した検査と実レンダーの到達検査は別の証拠であり、知覚的な再認や実音回収精度を主張しない。
- 設定の依存、尺度・期間・容量・cadence、不明キー、TOML往復を検査する。
- EOF直前の配送欠落と未配送のepoch変更をobserver経路で検査する。元支持を保持したまま入力終端で
  期限を判定し、旧epochの記憶や期限切れの結果を公開しない。
- 最終全テスト: **912成功・0失敗・28ignore**。`cargo test -- --nocapture`終了コード0、
  2026-09-12 22:57:45 JST。`RUST_BACKTRACE=1`、stdoutとstderrの両方を保存した。
- 修正後の`cargo clippy -- -D warnings`、`cargo check --all-targets`、`cargo fmt --all -- --check`、
  `git diff --check`を通過。I3・I4の技術実装の終了条件を満たした。

I3の既存frontend・lifecycle・feature数値fixtureも全テストで再検証する。
コード変更後の全出力はルートの`test_report.txt`、終了コードと時刻は`test_status.txt`に保存する。

## 残る範囲

UIとreportは同じsnapshotを消費する。実画面の見切れ・操作性、実機音声、長時間遅延、全負荷、fit、
聴取課題、可聴性、作者採用は未検証。追加のSpan確保、prefixコピー、matcher配列・path、snapshotと
queueの費用は専用observer内にも存在する。音声callbackや主workerへ検索処理を追加していないが、
全worker同時負荷のR2・実時間受入A3の合格を意味しない。

次はI5「gestureと受け渡し」。I4の区間診断をそのままgestureやphraseへ改名しない。
EOFの未確定短区間、長期記憶の採用方策、fit／保存codec交換を含むMR3全体、全曲文脈は後続に残す。
EOFでは発行済み結果を排出するが、途中の診断区間を完成区間へ昇格させない。
