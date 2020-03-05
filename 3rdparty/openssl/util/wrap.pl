#! /usr/bin/env perl

use strict;
use warnings;

use File::Basename;
use File::Spec::Functions;

my $there = canonpath(catdir(dirname($0), updir()));
my $std_engines = catdir($there, 'engines');
my $std_providers = catdir($there, 'providers');
my $unix_shlib_wrap = catfile($there, 'util/shlib_wrap.sh');

$ENV{OPENSSL_ENGINES} = $std_engines
    if ($ENV{OPENSSL_ENGINES} // '') eq '' && -d $std_engines;
$ENV{OPENSSL_MODULES} = $std_providers
    if ($ENV{OPENSSL_MODULES} // '') eq '' && -d $std_providers;

my $use_system = 0;
my @cmd;

if (($ENV{EXE_SHELL} // '') ne '') {
    # We don't know what $ENV{EXE_SHELL} contains, so we must use the one
    # string form to ensure that exec invokes a shell as needed.
    @cmd = ( join(' ', $ENV{EXE_SHELL}, @ARGV) );
} elsif (-x $unix_shlib_wrap) {
    @cmd = ( $unix_shlib_wrap, @ARGV );
} else {
    # Hope for the best
    @cmd = ( @ARGV );
}

# The exec() statement on MSWin32 doesn't seem to give back the exit code
# from the call, so we resort to using system() instead.
my $waitcode = system @cmd;

# According to documentation, -1 means that system() couldn't run the command,
# otherwise, the value is similar to the Unix wait() status value
# (exitcode << 8 | signalcode)
die "wrap.pl: Failed to execute '", join(' ', @cmd), "': $!\n"
    if $waitcode == -1;
exit($? & 255) if ($? & 255) != 0;
exit($? >> 8);
