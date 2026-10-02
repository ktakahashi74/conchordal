# F3a: 有界なsource別解析状態の参照実装

日付: 2026-09-26。状態: 実装前登録。F1/F2の解析を最大4 sourceへまとめる実験用処理器。
通常runtime配線、pitch採用、queue・実時間資源の合格は含まない。
関連: [全体計画](body-aware-fitness-plan.md)、[runtime境界](body-aware-fitness-runtime-handoff.md)。

## 入力と所有

処理器は観測開始時にresetした共有AnalysisStream一つと、追跡対象のsource別AnalysisStreamを最大4個所有する。
既定48 kHzの構築経路を注入できる形とし、数値単体試験は先行F1の8 kHz設定でも行う。
この単位では解析設定・epoch・sample rate・hop・周波数格子は固定する。設定変更は別処理器の新しい観測epochとする。

一回の入力は同じframeの混合habitat PCM、完全な追跡対象リスト、各対象の実際のhabitat PCMとする。
source id／source generation／出生sampleを付ける。休符はリストから消さず、全長のゼロPCMを渡す。
リストからの削除は追跡の終了であり、退役音のtailは混合PCMに残る。
新規登録は出生sampleと当該frame開始が一致する場合だけ認め、共有解析を当該frameの処理前に複製する。
同じidの異なる世代が同時に混在する入力は、この最初の処理器ではunsupportedとする。
実際の呼出し側には、新sourceが鳴る前の登録とTone所属の正しい採取が別途必要である。

新sourceを除いた過去が共有解析に保存されているので、新生sourceは過去を無音で埋めず開始できる。
既存sourceの途中登録は拒否する。終了済みsourceを同じ出生sampleで再登録しても拒否する。
共有解析は入力ごとに一回、各source解析は混合−自己PCMで一回処理する。
代表身体密度、候補生成、habituationの履歴はこの処理器へ混ぜない。

## 境界と欠測

frame番号は観測起点から連続し、入力PCMは全て正しいhop長で有限値を持つ必要がある。
sourceの重複、容量超過、出生時刻の不一致、epoch違い、長さ不一致、不正PCM、frame欠落・逆行を理由別に拒否する。
無効frameを読み飛ばして過去の状態を有効と報告しない。拒否後は当該処理器を無効とし、明示した新しい観測起点で
処理器を作り直すまで結果を出さない。有限warmupで欠落前の完全履歴を回復したとは扱わない。
新観測起点への切替を旧epochの継続として報告しない。

入力全体を検査してからsource状態を更新する。source容量は4で固定し、追跡順で無言に5番目を落とさない。
再利用するPCM scratchは一つのhop長に固定する。出生時の解析状態複製と解析器内部のallocationは
worker側の費用として後の資源計測へ残す。audio callbackへこの処理を置かない。

## 出力

出力にはepoch、source identity、支持終端sample、当該自己除去Landscapeを対応づける。
LandscapeのCは注入された同じ固定paramsで再計算し、共有habituationはまだ適用しない。
同じ入力frameの全source結果は同じ支持終端を持つ。一判断へ違う支持区間の結果を混ぜない。
この純粋な処理器は受信時刻や完了時刻を推測しない。後のworker wrapperと消費側が実際の発行・完了・受信時計を付ける。
frame支持終端をavailabilityの代わりとして使わない。

## 検査と次の境界

1. Sine/Harmonic/Modalを含む1 source／4 sourceで、各出力をそのsourceだけを除いた独立合成・独立解析と比較する。
   PCM・密度・H/R/CはF1と同じ許容を使う。単独除去は厳密なゼロ、同音他者は保存する。
2. 中途の新生source、退役sourceのtail、同時に複数が出生する場面を比較する。slot再利用で旧解析状態を渡さない。
3. 容量超過、出生より遅い登録、重複identity、epoch違い、欠落／逆行frame、不正hop・非有限PCMを拒否し、
   後続frameで結果が自然復活しないことを検査する。新しい処理器での明示的な観測開始と区別する。
4. 48 kHzの通常解析で4 sourceの固定場面を照合する。所要時間は参考値であり、並列取得中に実時間資源の合否を出さない。

通常解析の統合参照ではruntimeの `build_analysis_runtime_core(AppConfig::default(),48000)` を直接使う。
4 sourceはSine 440 Hz、Harmonic 440 Hz、Modal 466 Hz、Sine 660 Hz、各振幅0.35とし、
v3と同じhold=48000・attack=5 ms・release=0.5秒・SeqGate=1秒・brightness=0.8で64 hop観測する。
混合renderer一つ、自己-only renderer四つ、各sourceを除く独立renderer四つを使い、
processor出力をsourceごとの他者-only解析へ照合する。core層で別に行う手作りの倍音／非調和PCMによる
状態遷移試験は、この実Tone・runtime構築の検査と区別する。

次のF3bで、bounded queue、frame bufferの再利用、結果配送、ageとbody/candidate identityの検査を登録する。
共有runtimeへの配線はI4の境界確定後とし、現在のA4基準版と既定動作は維持する。
