# 身体評価の統合第九版: 取得前登録

2026-09-27。統合第八版を保存したまま、次の三単位を隔離worktree `.worktrees/integration-fitness` へ取り込む。

- F3e第四段階: 密度準備中の提案延期と、延期時間を一度だけ反映する時計の修正。試験専用。
- F4d: Random・Hereditary出生のsettle候補検査。試験専用。
- 通常runtimeの自己除去観測: `body_fitness_observation = true` の明示指定時だけ実音を解析し、reportへ観測結果を出す。行動への身体評価入力は変更しない。

元worktreeの記録とsource hashを保存し、統合前に第八版のsource manifestとの一致を検証する。更新時刻を引き継がずコピーし、共通ファイルは差分を比較して双方の変更を保持する。src・tests・Cargoファイルを取得前後にSHA256で照合する。

全cargo testをbacktrace・全出力付きで実行し、同じshellで終了値を記録する。新規の提案延期、F4d、通常render観測の検査名が実行ログにあることを確認する。fmt、標準Clippy、全target check、release buildを実行する。all-targets Clippyは今回の必須項目へ追加しない。

通常releaseの登録済み固定4条件（I4 bounded Sineと、I4/I11両ONのSine・Harmonic・Modal）を、第八版と同じ入力・第六版との比較器で取得する。観測設定は従来どおり省略し、既定OFFでのWAV byte、主要記録、非診断temporal、共通候補、成功CDF、実消費集計を照合する。render binaryが第八版とbyte一致する場合のみ既存4条件へ対応づける。異なる場合は全4条件を再取得し、良い結果だけを選び直さない。

通常render観測のON/OFFは個別登録の1/4 source・容量超過負例で確認する。live-paced配送は別登録・別の専有時間帯で取得し、この統合のoffline renderを実時間性能へ読み替えない。途中出生・退役の検査を追加する場合も取得前の個別登録を保存する。

既存のModal記憶参照0、I11資源2条件の未達、部分記憶の厳格OFF比較の未達は独立した未解決事項として保持する。通常動作の身体評価採用、音色遺伝、作者採用、実device資源受入は第九版の完了に含めない。
