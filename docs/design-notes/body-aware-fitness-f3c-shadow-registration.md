# F3c: 通常runtimeから自己除去環境までの観測接続

日付: 2026-09-26。状態: 取得前登録。静的実装済み、ビルド・実行は未検査。F3bの単体配送試験が通った後に行う実runtime参照検査。
通常の `wire_runtime` を通し、実VoiceとScheduleRendererから自己PCMを採取する。
移動や代謝の評価式はまだ変更しない。既定採用と実時間資源の検査ではない。

## 対象と出所

48 kHz・hop512・通常の解析構築を固定する。固定seed、出生時から1／4 sourceを観測する短いsceneを使い、
Sine／Harmonic／Modal、同音の他者を含める。自己音はScheduleRendererがsource単位・routing後に保存した
habitat PCMを使う。現在Voiceの基音や宣言スペクトルから自己音を再構成しない。
既存の `offline_body_probe` 接続点と、テスト専用のread-onlyな自己PCM参照だけを追加する。
捕捉が開始されていないsourceを後から「出生時から有効」と扱わない。

取得するsceneはseed 1／4とsource数1／4の四条件。1 sourceはSine 440 Hz、4 sourceは
Sine 440 Hz・Harmonic 440 Hz・Modal 466 Hz・Sine 660 Hzの順に出生させる。
全Voiceはamp 0.06、Seq brain、sustain、anchor、sustain drive 0.2、ADSR
(0.005, 0.0, 1.0, 0.2)とし、habitat／presentation両busへ送る。出生時刻0、generation 0、
source id 1から連番、scene終了0.55秒。最初の48 hopを照合する。habituationとDCC couplingは無効。
Rhaiからscenarioを構築し、実Voiceのid・generation・身体と最初の同音二者の基音を検査する。

支持区間は実runtimeが生成したPCMのframe start/end。source idとgenerationは実Voiceから取り、
この固定sceneでは最初に観測したframeを出生frameと照合する。SourceInputをF3bのworkerへ提出し、
受信・採用まで通す。retire後の音は混合環境に残す。source数が4を超えたら無言に選別せず失敗させる。

## offline配送と比較

この検査は明示したoffline配送を使う。frameを生成し、そのframeの解析結果が実threadを通って
受信されてから次frameへ進む。生成側のcommit時計をframe endとし、提出・受信・判断のsample時計を
このoffline commitに合わせる。実threadの処理開始・完了Instantは別に保持する。
待機はテスト経路に限り、通常音声callbackや通常runtimeへ待機を入れない。
この方式でageを満たしても、実時間の100 ms期限を満たした証拠にはならない。

同じ実PhonationBatch列を使い、各sourceを除いた独立ScheduleRendererと独立AnalysisStreamを起点から動かす。
捕捉PCMを混合から引いた値が他者-only PCMと一致し、配送後の密度とH/R/Cが独立参照とF1の許容内で一致するか検査する。
単独source除去はraw密度0、同音他者は保存、source identity・支持終端・解析epochも一致を必須とする。

追加の自己除去workerを無効にした同じsceneも走らせ、両busの音声がbit一致することを確認する。
固定音sceneだけの観測非干渉の検査であり、全シナリオ・全学習recordの回帰と呼ばない。
worker結果を移動へ使わず、参照件数0で音声だけ一致した条件を成功としない。
有効な受信・独立参照照合が最低24 hop継続したことと、数値誤差の最大値を保存する。

## 実施順

I11 §5.7専有取得中は静的実装だけ行う。F3b単体試験と代表密度の最適化検査を終えた版で、
この通常runtime経路を検査する。失敗時は源音・捕捉・配送のどこで差が出たか記録し、許容を事後変更しない。
新しい通常設定・Rhaiキー・公開technoteは追加しない。成功後も、候補身体の計算と移動への採用は次の単位として残す。
