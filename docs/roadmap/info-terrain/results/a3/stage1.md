# A-3 stage 1

Evaluation: [10,20) s. Cross-talk limit: 0.6 log10 units. Values: at least 7/9 within 0.5 log10 units.

| Variant | Path | rho v | rho s | value v | value s | max s cross-talk | max v cross-talk | min valid-hop rate | Pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| base | direct | Some(0.8432740427115678) | Some(0.9486832980505138) | 6/9 | 9/9 | Some(0.42087077448080645) | Some(0.35861867737567765) | 1.000000 | false |
| base | analysis | Some(0.4216370213557839) | Some(0.8432740427115678) | 5/9 | 3/9 | Some(0.3635413007878552) | Some(0.9788341678751706) | 1.000000 | false |
| seed+1000 | direct | Some(0.7905694150420948) | Some(0.9486832980505138) | 6/9 | 9/9 | Some(0.2639481226707612) | Some(0.3850154149487812) | 1.000000 | false |
| seed+1000 | analysis | Some(0.4743416490252569) | Some(0.8432740427115678) | 5/9 | 3/9 | Some(0.2120943127921655) | Some(1.0198144675374174) | 1.000000 | false |
| seed+2000 | direct | Some(0.7905694150420948) | Some(0.9486832980505138) | 6/9 | 9/9 | Some(0.4122417789397139) | Some(0.715127915333857) | 1.000000 | false |
| seed+2000 | analysis | Some(0.4743416490252569) | Some(0.7905694150420948) | 5/9 | 3/9 | Some(0.17417934546027514) | Some(0.905275317379826) | 1.000000 | false |
| particles128 | direct | Some(0.6851601597031488) | Some(0.9486832980505138) | 6/9 | 9/9 | Some(0.3396679152957005) | Some(0.46574234331659703) | 1.000000 | false |
| particles128 | analysis | Some(0.4216370213557839) | Some(0.9486832980505138) | 5/9 | 3/9 | Some(0.11531324065815518) | Some(1.125109277821724) | 1.000000 | false |
| initial | direct | Some(0.9486832980505138) | Some(0.9486832980505138) | 6/9 | 6/9 | Some(0.39941241095622804) | Some(0.4254897554466013) | 1.000000 | false |
| initial | analysis | Some(0.6851601597031488) | Some(0.7905694150420948) | 5/9 | 3/9 | Some(0.32817856287949754) | Some(0.6790650713713857) | 1.000000 | false |

| Variant | Path | seed | true v | true s | median v | median s | v value pass | s value pass | valid hops / total | rate pass |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|
| base | direct | 101 | 0.0001 | 0.001 | Some(0.00119868623793411) | Some(0.000975874708455016) | false | true | 937/937 | true |
| base | direct | 102 | 0.0001 | 0.01 | Some(0.0009972249875032269) | Some(0.009460438487300436) | false | true | 937/937 | true |
| base | direct | 103 | 0.0001 | 0.1 | Some(0.0022772558661345774) | Some(0.08815799031461041) | false | true | 937/937 | true |
| base | direct | 104 | 0.001 | 0.001 | Some(0.0013346712764238952) | Some(0.0012561658295156953) | true | true | 937/937 | true |
| base | direct | 105 | 0.001 | 0.01 | Some(0.0013688257223458719) | Some(0.009219348267771772) | true | true | 937/937 | true |
| base | direct | 106 | 0.001 | 0.1 | Some(0.0026271416299659773) | Some(0.0972832940600479) | true | true | 937/937 | true |
| base | direct | 107 | 0.01 | 0.001 | Some(0.006745170263514408) | Some(0.0025719637137775002) | true | true | 937/937 | true |
| base | direct | 108 | 0.01 | 0.01 | Some(0.008604722903654025) | Some(0.010836343279619624) | true | true | 937/937 | true |
| base | direct | 109 | 0.01 | 0.1 | Some(0.011599994805553078) | Some(0.09127854567692899) | true | true | 937/937 | true |
| base | analysis | 101 | 0.0001 | 0.001 | Some(0.0008927833828501588) | Some(0.0007722678725385387) | false | true | 937/937 | true |
| base | analysis | 102 | 0.0001 | 0.01 | Some(0.001807234714239152) | Some(0.0012066826587729845) | false | false | 937/937 | true |
| base | analysis | 103 | 0.0001 | 0.1 | Some(0.008503158336956399) | Some(0.003046061784773579) | false | false | 937/937 | true |
| base | analysis | 104 | 0.001 | 0.001 | Some(0.0014844331251542113) | Some(0.0010781061485036025) | true | true | 937/937 | true |
| base | analysis | 105 | 0.001 | 0.01 | Some(0.0015548376679133575) | Some(0.000894900232242057) | true | false | 937/937 | true |
| base | analysis | 106 | 0.001 | 0.1 | Some(0.010773714105605948) | Some(0.003031405193763192) | false | false | 937/937 | true |
| base | analysis | 107 | 0.01 | 0.001 | Some(0.003285209852681844) | Some(0.0010700146289753397) | true | true | 937/937 | true |
| base | analysis | 108 | 0.01 | 0.01 | Some(0.004553040238649398) | Some(0.0020668831349297603) | true | false | 937/937 | true |
| base | analysis | 109 | 0.01 | 0.1 | Some(0.013911782563603672) | Some(0.0022974458860422057) | true | false | 937/937 | true |
| seed+1000 | direct | 1101 | 0.0001 | 0.001 | Some(0.001052988980361245) | Some(0.0016487627679948322) | false | true | 937/937 | true |
| seed+1000 | direct | 1102 | 0.0001 | 0.01 | Some(0.0013510409770853735) | Some(0.008355837726918294) | false | true | 937/937 | true |
| seed+1000 | direct | 1103 | 0.0001 | 0.1 | Some(0.0025552843860004947) | Some(0.0933761694932487) | false | true | 937/937 | true |
| seed+1000 | direct | 1104 | 0.001 | 0.001 | Some(0.0013566678092078403) | Some(0.001491611538938417) | true | true | 937/937 | true |
| seed+1000 | direct | 1105 | 0.001 | 0.01 | Some(0.0015641551466261284) | Some(0.009587479834205222) | true | true | 937/937 | true |
| seed+1000 | direct | 1106 | 0.001 | 0.1 | Some(0.0023087384381806774) | Some(0.09297354851278994) | true | true | 937/937 | true |
| seed+1000 | direct | 1107 | 0.01 | 0.001 | Some(0.006968417806491802) | Some(0.0027390745772583945) | true | true | 937/937 | true |
| seed+1000 | direct | 1108 | 0.01 | 0.01 | Some(0.009818447166128832) | Some(0.01038886258822709) | true | true | 937/937 | true |
| seed+1000 | direct | 1109 | 0.01 | 0.1 | Some(0.012121749062238037) | Some(0.0969753123050577) | true | true | 937/937 | true |
| seed+1000 | analysis | 1101 | 0.0001 | 0.001 | Some(0.0009773304251576025) | Some(0.001052490384623531) | false | true | 937/937 | true |
| seed+1000 | analysis | 1102 | 0.0001 | 0.01 | Some(0.0014784505561308585) | Some(0.0017251001548187514) | false | false | 937/937 | true |
| seed+1000 | analysis | 1103 | 0.0001 | 0.1 | Some(0.010229534855917053) | Some(0.0024294614645271693) | false | false | 937/937 | true |
| seed+1000 | analysis | 1104 | 0.001 | 0.001 | Some(0.0015522906984286448) | Some(0.0014880435160503518) | true | true | 937/937 | true |
| seed+1000 | analysis | 1105 | 0.001 | 0.01 | Some(0.0016950841714602064) | Some(0.0015955698534596153) | true | false | 937/937 | true |
| seed+1000 | analysis | 1106 | 0.001 | 0.1 | Some(0.009239276076262855) | Some(0.0028745516040350628) | false | false | 937/937 | true |
| seed+1000 | analysis | 1107 | 0.01 | 0.001 | Some(0.003427110942150203) | Some(0.0017151908439999965) | true | true | 937/937 | true |
| seed+1000 | analysis | 1108 | 0.01 | 0.01 | Some(0.005156062954484469) | Some(0.0016518634995804327) | true | false | 937/937 | true |
| seed+1000 | analysis | 1109 | 0.01 | 0.1 | Some(0.01457452418311799) | Some(0.002599838533959383) | true | false | 937/937 | true |
| seed+2000 | direct | 2101 | 0.0001 | 0.001 | Some(0.0005969009239071726) | Some(0.001131441136979689) | false | true | 937/937 | true |
| seed+2000 | direct | 2102 | 0.0001 | 0.01 | Some(0.0011018248633528553) | Some(0.00901318010146189) | false | true | 937/937 | true |
| seed+2000 | direct | 2103 | 0.0001 | 0.1 | Some(0.00309763445592789) | Some(0.09066014117609382) | false | true | 937/937 | true |
| seed+2000 | direct | 2104 | 0.001 | 0.001 | Some(0.0016404393059179846) | Some(0.0018546395421214217) | true | true | 937/937 | true |
| seed+2000 | direct | 2105 | 0.001 | 0.01 | Some(0.0013031218491544052) | Some(0.009598398780181237) | true | true | 937/937 | true |
| seed+2000 | direct | 2106 | 0.001 | 0.1 | Some(0.0029117632109161794) | Some(0.0885871664591726) | true | true | 937/937 | true |
| seed+2000 | direct | 2107 | 0.01 | 0.001 | Some(0.00782551351401421) | Some(0.002923302404062262) | true | true | 937/937 | true |
| seed+2000 | direct | 2108 | 0.01 | 0.01 | Some(0.009704411266773866) | Some(0.009232250095125646) | true | true | 937/937 | true |
| seed+2000 | direct | 2109 | 0.01 | 0.1 | Some(0.008604961005838713) | Some(0.10177850154202728) | true | true | 937/937 | true |
| seed+2000 | analysis | 2101 | 0.0001 | 0.001 | Some(0.0012378340505679567) | Some(0.0014088114628846595) | false | true | 937/937 | true |
| seed+2000 | analysis | 2102 | 0.0001 | 0.01 | Some(0.0013039320312887546) | Some(0.0017877508410052833) | false | false | 937/937 | true |
| seed+2000 | analysis | 2103 | 0.0001 | 0.1 | Some(0.009952627331589746) | Some(0.002426426855405394) | false | false | 937/937 | true |
| seed+2000 | analysis | 2104 | 0.001 | 0.001 | Some(0.0012540371303043207) | Some(0.0011005025902446628) | true | true | 937/937 | true |
| seed+2000 | analysis | 2105 | 0.001 | 0.01 | Some(0.001957030107580368) | Some(0.0011970923227032349) | true | false | 937/937 | true |
| seed+2000 | analysis | 2106 | 0.001 | 0.1 | Some(0.009965993871286478) | Some(0.0027247885842938573) | false | false | 937/937 | true |
| seed+2000 | analysis | 2107 | 0.01 | 0.001 | Some(0.003232375240664112) | Some(0.0016134399359928372) | true | true | 937/937 | true |
| seed+2000 | analysis | 2108 | 0.01 | 0.01 | Some(0.004416559450320957) | Some(0.0015017165881041658) | true | false | 937/937 | true |
| seed+2000 | analysis | 2109 | 0.01 | 0.1 | Some(0.014343307209935992) | Some(0.002579500498268806) | true | false | 937/937 | true |
| particles128 | direct | 101 | 0.0001 | 0.001 | Some(0.0006671292012013512) | Some(0.0010075868818901139) | false | true | 937/937 | true |
| particles128 | direct | 102 | 0.0001 | 0.01 | Some(0.0014163867400742362) | Some(0.00902301344737006) | false | true | 937/937 | true |
| particles128 | direct | 103 | 0.0001 | 0.1 | Some(0.001949630427045538) | Some(0.0884041913272205) | false | true | 937/937 | true |
| particles128 | direct | 104 | 0.001 | 0.001 | Some(0.0013318121675351799) | Some(0.0011080262263087185) | true | true | 937/937 | true |
| particles128 | direct | 105 | 0.001 | 0.01 | Some(0.0013766594681005918) | Some(0.009243688713414567) | true | true | 937/937 | true |
| particles128 | direct | 106 | 0.001 | 0.1 | Some(0.001890381496521656) | Some(0.0976481041320509) | true | true | 937/937 | true |
| particles128 | direct | 107 | 0.01 | 0.001 | Some(0.007668318502262509) | Some(0.00220267498620946) | true | true | 937/937 | true |
| particles128 | direct | 108 | 0.01 | 0.01 | Some(0.008748595751732008) | Some(0.01031276320574782) | true | true | 937/937 | true |
| particles128 | direct | 109 | 0.01 | 0.1 | Some(0.013142968346931008) | Some(0.09084528712545055) | true | true | 937/937 | true |
| particles128 | analysis | 101 | 0.0001 | 0.001 | Some(0.0008920245728495573) | Some(0.0007520655475779774) | false | true | 937/937 | true |
| particles128 | analysis | 102 | 0.0001 | 0.01 | Some(0.0012156910853746191) | Some(0.001046364854543996) | false | false | 937/937 | true |
| particles128 | analysis | 103 | 0.0001 | 0.1 | Some(0.00953875252360927) | Some(0.00210137433158996) | false | false | 937/937 | true |
| particles128 | analysis | 104 | 0.001 | 0.001 | Some(0.000874920365031898) | Some(0.0007640260416430127) | true | true | 937/937 | true |
| particles128 | analysis | 105 | 0.001 | 0.01 | Some(0.0014198060024494022) | Some(0.0009631879891813723) | true | false | 937/937 | true |
| particles128 | analysis | 106 | 0.001 | 0.1 | Some(0.011670186682336856) | Some(0.0022767284676236295) | false | false | 937/937 | true |
| particles128 | analysis | 107 | 0.01 | 0.001 | Some(0.0034391759444258405) | Some(0.0008554222582216458) | true | true | 937/937 | true |
| particles128 | analysis | 108 | 0.01 | 0.01 | Some(0.0038503191810581676) | Some(0.0012485088824360764) | true | false | 937/937 | true |
| particles128 | analysis | 109 | 0.01 | 0.1 | Some(0.014046070116544053) | Some(0.0017458141019866145) | true | false | 937/937 | true |
| initial | direct | 101 | 0.0001 | 0.001 | Some(0.0006431269625374944) | Some(0.0032102498744701956) | false | false | 937/937 | true |
| initial | direct | 102 | 0.0001 | 0.01 | Some(0.0007800702738605691) | Some(0.010523511802137127) | false | true | 937/937 | true |
| initial | direct | 103 | 0.0001 | 0.1 | Some(0.0005919928617874906) | Some(0.09006604731507534) | false | true | 937/937 | true |
| initial | direct | 104 | 0.001 | 0.001 | Some(0.0007901064156982719) | Some(0.004004281649072204) | true | false | 937/937 | true |
| initial | direct | 105 | 0.001 | 0.01 | Some(0.00114886540759082) | Some(0.010674962181118611) | true | true | 937/937 | true |
| initial | direct | 106 | 0.001 | 0.1 | Some(0.0012405037429239433) | Some(0.09678394922779189) | true | true | 937/937 | true |
| initial | direct | 107 | 0.01 | 0.001 | Some(0.0038220062558351146) | Some(0.008052880391593034) | true | false | 937/937 | true |
| initial | direct | 108 | 0.01 | 0.01 | Some(0.004991741128298952) | Some(0.01386711812691077) | true | true | 937/937 | true |
| initial | direct | 109 | 0.01 | 0.1 | Some(0.010180782257218449) | Some(0.09217978130134928) | true | true | 937/937 | true |
| initial | analysis | 101 | 0.0001 | 0.001 | Some(0.0006415118350578229) | Some(0.006219053045843363) | false | false | 937/937 | true |
| initial | analysis | 102 | 0.0001 | 0.01 | Some(0.0011286933695621702) | Some(0.005611787862990116) | false | true | 937/937 | true |
| initial | analysis | 103 | 0.0001 | 0.1 | Some(0.002378295726931536) | Some(0.013809723402682793) | false | false | 937/937 | true |
| initial | analysis | 104 | 0.001 | 0.001 | Some(0.0005703771077049012) | Some(0.006056653866223105) | true | false | 937/937 | true |
| initial | analysis | 105 | 0.001 | 0.01 | Some(0.0009496679437930456) | Some(0.007323634291600131) | true | true | 937/937 | true |
| initial | analysis | 106 | 0.001 | 0.1 | Some(0.0027241257911792744) | Some(0.014156936261742498) | true | false | 937/937 | true |
| initial | analysis | 107 | 0.01 | 0.001 | Some(0.0032248739261801986) | Some(0.006864195467736275) | true | false | 937/937 | true |
| initial | analysis | 108 | 0.01 | 0.01 | Some(0.0028787806076738497) | Some(0.011947576179587846) | false | true | 937/937 | true |
| initial | analysis | 109 | 0.01 | 0.1 | Some(0.005710300229445208) | Some(0.021593920433562227) | true | false | 937/937 | true |
