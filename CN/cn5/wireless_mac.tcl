set chan [new $val(chan)]

$ns node-config \
    -adhocRouting $val(rp) \
    -llType $val(ll) \
    -macType $val(mac) \
    -ifqType $val(ifq) \
    -ifqLen $val(ifqlen) \
    -antType $val(ant) \
    -phyType $val(netif) \
    -channel $chan \
    -propType $val(prop) \
    -topoInstance $topo \
    -agentTrace ON \
    -routerTrace ON \
    -macTrace ON

set ns [new Simulator]

set topo [new Topography]
$topo load_flatgrid 500 500

create-god $val(nn)

$ns node-config \
    -adhocRouting $val(rp) \
    -llType $val(ll) \
    -macType $val(mac) \
    -ifqType $val(ifq) \
    -phyType $val(netif) \
    -channelType $val(chan) \
    -propType $val(prop) \
    -topoInstance $topo \
    -agentTrace ON

for {set i 0} {$i < $val(nn)} {incr i} {
    set node_($i) [$ns node]
}

$ns at 1.0 "$node_(0) setdest 250 250 20"
$ns at 2.0 "$node_(1) setdest 100 100 20"

$ns at 10.0 "exit 0"

$ns run