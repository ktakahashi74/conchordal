# 身体評価の通常runtime観測接続: 検証記録

2026-09-27。[取得前登録](body-fitness-runtime-observation-registration-20260927.md)に従い、統合第八版から隔離した `.worktrees/body-fitness-runtime` に実装し、統合第九版へ取り込んだ。TOMLルートの `body_fitness_observation = true` を明示した場合だけ動く。省略時はfalse。身体評価による移動・代謝・出生の選択変更はこの接続に含まない。

共有runtimeの実ScheduleRendererから、同じhopの混合habitat PCMと各Voiceの自己habitat PCMを取得し、既存の有界source-removed workerへ渡す。48 kHz、512 sample/hop、最大4 sourceに限定する。instrumentの受信はnonblocking、offline renderだけが明示的に完了を待つ。reportに配送modeを記す。workerのjoinはruntime終了時で、hop内では行わない。ただしworker生成と解析状態の初期化は最初の観測hopで同期実行するため、この起動費用は未評価である。

通常render binaryの1/4/5 sourceをON/OFFで比較した。1/4 sourceでは受理があり、1 sourceの自己除去massは0、4 sourceには他者の正massがある。支持・受信・判断の時刻順とage上限を検査した。全条件でWAV byteと登録した既存行動recordが一致し、5 sourceでは理由 `CapacityExceeded` と受理0を確認した。設定の既定OFFと、48 kHz/512以外の拒否も検査した。

独立レビューで、容量超過の負例が理由の文字列型だけを検査していた点と、今hopで未到着でも過去の受理値を区別せず表示できる点を発見した。前者を理由の完全一致へ修正した。後者は `accepted_this_hop`、`current_result_available`、欠測hop数を加え、source値・supported数・ageを最後の受理時点の情報として明示した。出力の脱落を `OutputDropped` で停止し、入力queue飽和の停止とは別に計数する。解析設定の変更も理由付きで停止し、自動再開しない。

レビュー前の全suiteは2026-09-27 11:43:58 JSTに1245成功・0失敗・43 ignore、exit 0。修正後の欠測単体と通常render比較は各1件通過した。最終sourceの全suite・fmt・Clippy・全target check・release buildは統合第九版の記録へ対応づける。途中出生と退役は[別登録](body-fitness-runtime-lifecycle-registration-20260927.md)で検査する。

通常ビルド経路での観測接続を示す結果である。実device性能、初回hopの費用、通常runtimeでの身体評価消費、有効評価率、音色遺伝、作者採用の証拠ではない。live配送の飽和が実際に何回発生するかは未取得であり、offline renderの受理率から推定しない。
