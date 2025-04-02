%%  Method 2. Generate a @cMosic object from scratch
%% Initialize
ieInit;

% Generating from scratch can be slow, especially
% if the mosaic eccentricity is off-axis
cm = cMosaic(...
    'size degs', [10 10], ...            % SIZE: x=0.5 degs, y=0.5 degs
    'position degs', [0.0 0.0], ...      % ECC:  x=1.0 degs, y= 0.0 degs
    'compute mesh from scratch', true, ...   % generate mesh on-line, will take some time
    'random seed', randi(9999999), ...     % set the random seed, so at to generate a different mosaic each time
    'max mesh iterations', 200 ...           % stop iterative procedure after this many iterations
    );

rows = size(cm, 1)  % Number of rows
cols = size(cm, 2)
conesnum = cm.conesNum
%% Visualize in a ieNewGraphWin

%%hFig = ieNewGraphWin;
%cm.visualize(...
   % 'figureHandle', hFig, ...
   % 'axesHandle', gca, ...
   % 'domain', 'degrees', ...
   % 'plotTitle', 'on-line mesh generation');

%drawnow;

all_cones_degrees = cm.coneRFpositionsDegs;
writematrix(all_cones_degrees,'/Users/maria/Desktop/ISETBioSimulations/Mosaic0/0_0/all_cones_degrees.csv');
all_cones_microns = cm.coneRFpositionsMicrons;
writematrix(all_cones_microns,'/Users/maria/Desktop/ISETBioSimulations/Mosaic0/0_0/all_cones_microns.csv')

cone_locations_microns = cm.coneApertureDiametersMicrons;
cone_locations_microns_transposed = cone_locations_microns';
writematrix(cone_locations_microns_transposed,'/Users/maria/Desktop/ISETBioSimulations/Mosaic0/0_0/cone_aperature_microns.csv');

cone_locations_degrees = cm.coneApertureDiametersDegs;
cone_locations_degrees_transposed = cone_locations_degrees';
writematrix(cone_locations_degrees_transposed,'/Users/maria/Desktop/ISETBioSimulations/Mosaic0/0_0/cone_aperature_degrees.csv');



save('/Users/maria/Desktop/ISETBioSimulations/Mosaic0/0_0/worksapce.mat')