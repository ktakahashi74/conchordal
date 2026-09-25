# I11第2段・登録入力

2026-09-23 PDT／24 JST。Astraレビュー後、実装前に凍結した入力。本文と判定規則の正本は[第2段登録](../i11-2-arrival-comparison.md)。凍結時点で`inputs/`の13ファイルと`plan.json`は作成・SHA照合したが、Rhaiの実行、CDF生成、render、profile取得は行っていない。入力・判定規則のSHAは`freeze.json`へ保存。候補binary・source・検査器は実装後かつ取得前に別manifestへ固定する。

## scenarioと対照

旧登録12条件はbyte不変で維持し、別の外部pulse 6条件を追加した。追加6条件はsine／harmonic／modal × 総4／16 Voice。各scriptは `temporal_mode("observe")`、seed `20260924`、6秒＋終端0.5秒、48 kHz configで共通。2 Hzの1声は `sine().brain("drone").repeat().pulse(2.0).cycles(1)` を880 Hz、habitat busだけへ送る。残る3／15声は110／220／330／440 Hzの元のflow方策をpresentation busだけへ送る。habitat observerに選択対象の自声が混ざらず、外部pulseと選択対象はbusで分離される。pulseの `PhonationTiming::Pulse` は `TemporalParticipation` のcoupled policyを使わない。対象Voiceの参加決定だけをT2作用として数える。音群への実際の周期認定は未測定であり、適格な群や選択差が無ければ未達。結果を見た素材差替えはしない。

primary Hazardは18係数 `[-2,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0]`、means 8個0、deviations 8個1、horizon 4.0秒。係数10はI6の位相cos項。48 kHzの20 msは960 sample、CDFは起点を含む201点。既存I6の単一horizon probability／integralは変えず、第2段専用APIで隣接20 ms区間を積分し累積する。2 Hz（周期0.5秒）の4秒全区間を既存 `integral` に一度に渡すと、最細16 panelが0.25秒となり、位相条件の周期/4＝0.125秒を満たせない。既存probabilityの `None` をCDFの拒否理由へ流用しない。

`config-constant-hazard-on.toml`はI6の定数2/秒の解析照合専用。係数0だけ `ln(expm1(2))` のf64書式値、他は0。`config-periodic-on.toml`はPeriodic段差の対照。T1作用は旧12条件の `body-off` 対 `proxy-off`、T2作用は新6条件の `body-on` 対 `body-off` とする。`proxy-on`は任意のfixture設定で、主取得には含めない。同一観測を凍結した単体介入では同じCDFをon/offへ渡し、到来費用だけを切り替える。実走ではCDFをonでのみ作るため、off側CDF欠損を許す。最初の分岐より前のschedule、候補構造、footprint record／power、外部予測、非到来状態はbit一致が必要。skip差だけが先に現れても、選択された候補のbase costでskipする規則に合えば正当な分岐として検査する。

## planとSHAの使い方

`plan.json`の`input_sha256`は今回の凍結bytes、`existing_stage1_inputs`は元12 scriptの登録pathとSHA。旧登録原本は改変していない。primary係数・means・deviations・horizonの全f64 bitもplanへ保存した。各configのprofile欄は旧登録binaryのSHA `033905eadd9a7c681a8568ea347c25d1c1482d49d708bc849b00607a85d51fb5`を維持し、取得時は`--profile-file`相当の方法で実行用コピーのpathだけを変更する。派生configは元とのparse比較とprofile SHAを検証し、両configのSHAを記録する。全script/config、本登録文書、検査器、Rust source、toolchain、baseline／候補binaryのSHAを**取得前**に別の凍結planへ確定する。原None基準は同Macの `i11-stage1-baseline`（tag object `9e4ed2b4d967b6232548a50adbb1002696ca794d`、peelしたcommit `103a32fe5c5fc4fdf0f4d65fb25a1bb0c7c47303`）、到来false回帰は保存済み第1段完了binary、T2作用は同じ第2段binaryのon/off設定で区別する。`baseline`には両基準manifestのpath・SHA、binary SHA、保存sourceのSHAを分けて記録した。候補版binaryと登録本文・検査器のSHAはまだ未確定。外部binaryを指定した取得で現checkoutのsource hashをそのbinaryの出所と誤記しない。

第2段の§5.4bは改訂済みの「対の再実行における共通キーの候補record内容一致（`processing_us`だけ除外）」を要件とする。片側のみのrecord件数は別集計。凍結packet再評価は未実施という限界を残すが、追加の合格条件にはしない。§5.4aは同じ新primary configを原None基準と第2段版に与え、旧12条件のWAVと学習10種をbit比較する。保存済み第1段のMac結果はHazard設定が違うため、body-off／proxy-offの旧12条件は同一新configで第1段binaryと第2段binaryを別取得して回帰比較する。旧12条件の第1段証拠は必要なbit回帰が通った範囲で再利用する。

## 取得行列と判定

offline renderの草稿行列は原None基準の旧12本、第1段完了版の旧12×body-off／proxy-off＝24本、第2段版の旧12×None／body-off／proxy-off／body-on＝48本、新6×body-off／body-on＝12本、計96本。§5.9はbody-onの旧12＋新6、計18条件の母集団別分布を報告する。§5.9の再投影は`representative_gap.rs`の規約どおり`issue.energy`をbus 0へ強制投影し、実routeで無音化しない。旧12と新6は別母集団に分け、新6の実routeをmetadataとして記録する。定数HazardとPeriodicは独立の核・配送・境界検査用で、主作用の素材差替えには使わない。

§5.7は旧8条件＋新pulse6条件＝14条件×report有／無の28組。各組でNone、body-off、body-onを交互に3反復し、body-on対Noneとbody-on対body-offの**両比較**を全10欄で判定する。同じMacでNone対NoneのA/Aを3 pass、各pass 28組×3反復取得し、各欄84差分の最大値をfloorとして先に凍結する。A/Aは504 run、本比較は252 run。各反復の統計から中央値を作る順と、超過hop数の反復合計も第1段の3 pass方式に合わせる。`population_us`、`synthesis_us`、生の`elapsed_us`各中央値・p99は `abs(差) <= max(0.05*対照値, floor)`、各最大は `abs(差) <= max(1000 us, floor)`、予算超過hop数の反復合計差はfloor以内。元登録のfloorなし±5%／最大1 ms／超過同数も併記する。新pulse6条件はreport有／無それぞれ全3反復のbody-on実行で、有効到来消費が各run 1件以上なければ性能合格扱いしない。no-reportの件数はprofileから判定する。A/A floorの不足が判明しても取得後に増やさず、未達または新登録とする。後の`own_us`への狭域化は第2段の総hop合格へ流用しない。

§5.1〜§5.3、§5.5、§5.6、§5.8の単体・独立参照は対象入力と出所、期限、非periodic、自己除外、積分失敗、候補全域のCDF、予約後非再選択を含む。受領時刻はruntimeが発行版を初めて得たsample clockで固定し、次hopで更新しない。到来false、weight 0、coupling 0、既知CDF 0、全群unknownを別fixtureにする。独立参照は登録本文の適応Simpson（絶対許容1e-11、最大深さ24）に従う。CDFの独立参照と最初の分岐では許容を混同しない。T2主作用は固定6条件で到来項だけによる最初の選択／skip分岐とWAV差を示し、3身体それぞれ少なくとも1条件が成立する。正例が無ければ不合格のまま残す。

`plan.json`の`run_order`はoffline96本→A/A504本→floor凍結gate→本比較252本の計852 runを1行ずつ記録した。A/Aはpass→case→report有無→repの順にNone-A/Bの先行を交替し、本比較の3設定はrepごとに順序を回転する。floorを保存・hash固定するまでは本比較を開始しない。他のrender、instrument、重いbuildを並列にしない。各runの終了コード、WAV／report／profileのSHA、binaryとsourceの出所をledgerへ残してから集計する。凍結時点ではこの実行を開始していない。
