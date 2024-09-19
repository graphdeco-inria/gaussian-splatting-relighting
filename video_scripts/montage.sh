
label="v6_no_key_views"

cat >montage.txt <<-EOF
file 'out_real_garagewall_${label}_ken_burns_1.mp4'
file 'out_real_garagewall_${label}_ken_burns_0.mp4'

file 'out_real_multilight_kettle_2_${label}_ken_burns_0.mp4'
file 'out_real_multilight_kettle_2_${label}_ken_burns_1.mp4'

file 'out_real_multilight_chestdrawer_2_${label}_ken_burns_1.mp4'
file 'out_real_multilight_chestdrawer_2_${label}_ken_burns_0.mp4'

file 'out_real_paintgun_${label}_ken_burns_1.mp4'
file 'out_real_paintgun_${label}_ken_burns_0.mp4'

file 'out_real_multilight_pans_2_${label}_ken_burns_1.mp4'
file 'out_real_multilight_pans_2_${label}_ken_burns_0.mp4'

file 'out_vendored_mipnerf_counter_${label}_ken_burns_1.mp4'
file 'out_vendored_mipnerf_counter_${label}_ken_burns_0.mp4'

file 'out_vendored_mipnerf_room_${label}_ken_burns_1.mp4'
file 'out_vendored_mipnerf_room_${label}_ken_burns_0.mp4'
EOF

ffmpeg -y -f concat -i montage.txt -c copy montage_$label.mp4

cat >montage_orbit.txt <<-EOF
file 'out_real_garagewall_${label}_orbit_0.mp4'
file 'out_real_multilight_kettle_2_${label}_orbit_0.mp4'
file 'out_real_multilight_chestdrawer_2_${label}_orbit_0.mp4'
file 'out_real_multilight_pans_2_${label}_orbit_0.mp4'
file 'out_real_paintgun_${label}_orbit_0.mp4'
file 'out_vendored_mipnerf_counter_${label}_orbit_0.mp4'
file 'out_vendored_mipnerf_room_${label}_orbit_0.mp4'
EOF

if [ -v CRF ]; then 
    flags="-crf $CRF"
else
    flags=""
fi
ffmpeg -y -f concat -i montage_orbit.txt -c copy $flags montage_orbit_$label.mp4