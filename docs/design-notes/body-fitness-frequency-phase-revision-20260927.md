# 周波数代表発音のdrone初期位相固定（2026-09-27）

状態: 旧条件の失敗観測後、修正版の実装・取得前に記録するモデル改訂。通常renderの非Sine登録に対する最初の取得では、frame48/96でHarmonicとModalのbody_generationが毎hop増え、評価消費が0だった。旧binaryとsourceは非Sine取得の `pre-revision-snapshot/` に保存し、最初の取得は旧binaryのまま継続する。

原因はI11の代表onset用 `representative_modulator` が、kick後も残る `DroneSway.phase` を保持することにある。これを周波数代表発音にそのまま使うと、固定身体でも毎hop Recipeが変わり、72-hopの候補計算が消費前に失効する。I11の契約を誤りとは扱わない。周波数側の計画§4.1は現励振状態から独立した代表発音を定義し、I11と同じ量であるとはしていない。

## 修正版の条件

`Voice::representative_body_recipe` に限り、DroneSwayの初期位相を0 radへ固定する。sway_rate、身体、ADSR、代表振幅、hold、72-hop観測は保持する。実際のrenderer、自己音除去用PCM、I11のrepresentative_modulatorとその位相identityは変更しない。これは位相に対する密度の不変性を発見したという主張ではなく、周波数比較用の代表条件の選択である。実演奏の瞬時位相を予測する用途には使わない。

8 kHzの小型数値試験では、実Voiceのdrone位相だけを変えて周波数Recipeと実Tone/NSGT由来の代表密度が同じになることを検査する。一方、実render modulatorとI11用identityでは位相の違いを保持する。sway_rateを0.05から0.4 Hzへ変えれば周波数Recipeも密度も変わり、身体brightness変更も引き続き失効要因となる。

修正後は既登録のseed17、Harmonic/Modal、drone、身体・control変更時刻、各シーンの長さを変えず、別ディレクトリで二回ずつ再取得する。旧失敗と修正版成功を同じ事前登録の成功としてまとめない。身体評価の因果効果、実時間費用、可聴差の作者評価は別の判定とする。

## 改訂後の検証

位相固定とsway_rate/brightnessの区別を検査する数値試験は通過した。通常renderでは位相による毎hop失効が解消した。その後、旧at配置がPitchMode::Lockを設定していた別のfixture問題を区別し、取得前にline配置へ改訂した[非Sine v2](body-fitness-runtime-changes-v2-registration-20260927.md)で実target・基音の変更を確認した。第十二版bの全体試験1271件と限定実時間条件も通過したが、位相固定が実演奏の瞬時状態を予測するという主張は追加しない。
