%%  Method 2. Generate a @cMosic object from scratch
%% Initialize
ieInit;


for mosaic_num=0:1:5
    
    % Generating from scratch can be slow, especially
    % if the mosaic eccentricity is off-axis
    cm = cMosaic(...
        'size degs', [10 10], ...            % SIZE: x=0.5 degs, y=0.5 degs
        'position degs', [0.0 0.0], ...      % ECC:  x=1.0 degs, y= 0.0 degs
        'compute mesh from scratch', true, ...   % generate mesh on-line, will take some time
        'random seed', randi(9999999), ...     % set the random seed, so at to generate a different mosaic each time
        'max mesh iterations', 100 ...           % stop iterative procedure after this many iterations
        );
    
    rows = size(cm, 1)  % Number of rows
    cols = size(cm, 2)
    conesnum = cm.conesNum

    
    all_cones_degrees = cm.coneRFpositionsDegs;
    formatted_string = sprintf('/Users/maria/Desktop/ISETBioSimulations/Mosaic%u/0_0/all_cones_degrees.csv', mosaic_num);
    writematrix(all_cones_degrees,formatted_string);

    all_cones_microns = cm.coneRFpositionsMicrons;
    formatted_string = sprintf('/Users/maria/Desktop/ISETBioSimulations/Mosaic%u/0_0/all_cones_microns.csv', mosaic_num);
    writematrix(all_cones_microns,formatted_string)
    
    cone_locations_microns = cm.coneApertureDiametersMicrons;
    cone_locations_microns_transposed = cone_locations_microns';
    formatted_string = sprintf('/Users/maria/Desktop/ISETBioSimulations/Mosaic%u/0_0/cone_aperature_microns.csv', mosaic_num);
    writematrix(cone_locations_microns_transposed,formatted_string);
    
    cone_locations_degrees = cm.coneApertureDiametersDegs;
    cone_locations_degrees_transposed = cone_locations_degrees';
    formatted_string = sprintf('/Users/maria/Desktop/ISETBioSimulations/Mosaic%u/0_0/cone_aperature_degrees.csv', mosaic_num);
    writematrix(cone_locations_degrees_transposed,formatted_string);
    


    save(sprintf('/Users/maria/Desktop/ISETBioSimulations/Mosaic%u/0_0/worksapce.mat', mosaic_num))

end 

