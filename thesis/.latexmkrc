use File::Basename;

add_cus_dep('glo', 'gls', 0, 'makeglossaries');
add_cus_dep('acn', 'acr', 0, 'makeglossaries');
add_cus_dep('ntn', 'not', 0, 'makeglossaries');

sub makeglossaries {
    my ($base_name, $path) = fileparse( $_[0] );
    return system "makeglossaries", "-d", $path, $base_name;
}

push @generated_exts, 'glo', 'gls', 'glg';
push @generated_exts, 'acn', 'acr', 'alg';
push @generated_exts, 'ntn', 'not', 'nlg';
$clean_ext .= ' %R.ist %R.xdy';