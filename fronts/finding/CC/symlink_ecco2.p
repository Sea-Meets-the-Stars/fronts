#!/usr/bin/perl

$nargs = $#ARGV + 1;
#print "#ARGs: '$nargs'\n";
#print "#ARGV[0]: '$ARGV[0]'\n";
#print "#ARGV[1]: '$ARGV[1]'\n";
#print "#ARGV[2]: '$ARGV[2]'\n";
#print "#ARGV[3]: '$ARGV[3]'\n";
#print "#ARGV[4]: '$ARGV[4]'\n";

$input_directory = $ARGV[0];
#print "Input Base_Directory is '$input_directory'\n";

$output_directory = $ARGV[1];
#print "Output Base Directory is '$output_directory'\n";

print "output directory layout is $output_directory/YYYY/MM/*\n";

$create_script = 0;

if ( $nargs < 4 ) {
    die "Usage: <input basedir> <output_basedir> <startdate> <enddate> [optional <script filename>]\n";
}

if ( $nargs == 5 ) {
    $output_file = $ARGV[4];
    $create_script = 1;
}

if ( -e $output_directory ) {

    if ( $create_script == 1 ) {
	open(OUTPUT_FILE, ">$output_file");
	print OUTPUT_FILE "#!/bin/sh\n\n";

	print "shell script will be written to $output_file\n";
    } else {
	print "symbolic links will be created under $output_directory\n";
    }

    print "Starting with Date: $ARGV[2] : Ending with Date: $ARGV[3]\n"; 
    my $string = $ARGV[2];   # Process Start Date

    $start_yr = substr($string,0,4);
    $start_mn = substr($string,4,2);
    $start_dy = substr($string,6,2);
    $startdate = $start_yr . $start_mn . $start_dy;
    
    if ( ! $startdate =~ /^[0-9]+$/) {
	die "Starting Date ($ARGV[2]) is not valid, format is YYYYMMDD with MM and DD being 2 characters wide.\n";
    }

    $string = $ARGV[3];   # Process End Date

    $end_yr = substr($string,0,4);
    $end_mn = substr($string,4,2);
    $end_dy = substr($string,6,2);
    $enddate = $end_yr . $end_mn . $end_dy;

    if ( ! $enddate =~ /^[0-9]+$/) {
	die "Ending Date ($ARGV[3]) is not valid, format is YYYYMMDD with MM and DD being 2 characters wide.\n";
    }

#    @dirlist = `ls -1 $input_directory`;

#    foreach my $dirline (@dirlist) {

#	chomp $dirline;
#	$subyear_created = 0;

#	@filelist = `ls -1 $input_directory/$dirline/`;
	@filelist = `ls -1 $input_directory/`;
		
	foreach my $fileline (@filelist) {
	    chomp $fileline;
		    
	    $name = $fileline;
	    $fullname = $name;	    
	    chop ($name);
	    $name =~ s/\W.*//;
	    
	    $prefix = substr($fullname,0,3);
	    $yr = substr($fullname,4,4);
	    $month = substr($fullname,8,2);
	    $day = substr($fullname,10,2);
	    #print "fileline: $fileline\n";
	    #print "fullname = '$fullname';name = '$name'; year = '$yr'; month = '$month'; day = '$day'; prefix = '$prefix'\n";
	    
	    $currentdate = $yr . $month . $day;
	    
	    if ($currentdate >= $startdate && $currentdate <= $enddate) {
		#print "prefix:$prefix \n";
		
		if ( $prefix eq 'SST') {  # Only use valid files for name decomposition values
		    
		    $output_subyear = $output_directory . '/' . $yr;
		    
		    if ( -e $output_subyear ) {
			$output_submonth = $output_directory . '/' . $yr . '/'. $month;
			if ( ! -e $output_submonth ) {
			    unless (mkdir $output_submonth) {
				die "Unable to create $output_submonth\n";
			    }
			}
		    } else {
			unless ( mkdir ($output_subyear) ) {
			    die "Unable to create $output_subyear\n";
			}
			
			$output_submonth = $output_directory . '/' . $yr . '/'. $month;
			
			unless (mkdir $output_submonth) {
			    die "Unable to create $output_submonth\n";
			}
		    }
		    
		    $input_file = $input_directory . '/' . $fullname;
		    $output_link = $output_submonth . '/' . $fullname;
		    if ( -e $output_link) {
			print "output file exists['$output_link'], not overwriting\n";
		    } else {
		      print "Writing['$input_file'], to['$output_link']\n";
			if ( $create_script == 1 ) {
			    print OUTPUT_FILE "ln -s $input_file $output_link\n";
			} else {
			    my $cmd = `ln -s $input_file $output_link`;
			}
		    }
		}
	    }
	}
#    }
} else {
    print "exiting: output base_directory does not exist: '$output_directory'\n";
}


     






