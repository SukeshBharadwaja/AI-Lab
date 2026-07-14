# Create Simulator
set ns [new Simulator]

# Trace File
set tracefile [open out.tr w]
$ns trace-all $tracefile

# NAM File
set namfile [open out.nam w]
$ns namtrace-all $namfile

# Create Nodes
set n0 [$ns node]
set n1 [$ns node]
set n2 [$ns node]
set n3 [$ns node]

# Create Star Topology
$ns duplex-link $n0 $n1 1Mb 10ms DropTail
$ns duplex-link $n0 $n2 1Mb 10ms DropTail
$ns duplex-link $n0 $n3 1Mb 10ms DropTail

# Finish Procedure
proc finish {} {
    global ns tracefile namfile
    $ns flush-trace
    close $tracefile
    close $namfile
    exit 0
}

$ns at 5.0 "finish"

$ns run