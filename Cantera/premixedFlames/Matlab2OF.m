%% Script to take data from MATLAB and convert to OpenFOAM
% 1D C2H4-air premixed flame initialized in Cantera using freeFlame solver

% flame conditions
    L = 0.2; % length of domain (m)
    Sl = 0.684668; % laminar flame speed (m/s) from Cantera
    Tb = 2387.204994; % adiabtic flame T or burnt gas T (K) from Cantera
    Tu = 300; % unburnt gas T (K)
    % maxdT_dx = 7153587.974117; % maximum of T gradient / grid gradient (K/m)
    % Lf = (Tb - Tu) ./ maxdT_dx ; % laminar flame thickness (m)
    % min_dx = Lf ./ 20;


cd('~/anaconda3/PROJECTS/premixedFlames');
tFolder = 0;
 
% read variable names from .csv file
    fname = 'c2h4_adiabatic.csv';   % Specify the file name to be opened 
    fid = fopen(fname); % Open file in read mode
        vars = textscan(fid,' %s ','Delimiter',',','MultipleDelimsAsOne',1);
    fclose(fid);        % Close the opened file    fid = fopen(fname); % Open file in read mode
        
    
% read data from Cantera .csv generated file  
    fid = fopen(fname); % Open file in read mode
        Cantera = textscan(fid,'%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f','Delimiter',',','headerlines',1,'EmptyValue',0);
    fclose(fid);        % Close the opened file
    N_data = length(Cantera{1});
    N_vars = length(Cantera) - 1;
    x_Cantera = Cantera{1};
     

% check laminar flame thickness calculation in Python with Matlab
    dx_Cantera = diff(Cantera{1});
    dT_Cantera = diff(Cantera{2});
    Lf = (Tb - Tu) ./ max(dT_Cantera./dx_Cantera) ;
    min_dx = Lf ./ 20;
    min_cells = L ./ min_dx;
    fprintf('Laminar flame speed: Sl = %.15g m/s \n',Sl)
    fprintf('Laminar flame thickenss: Lf = %.15g m \n',Lf)
    fprintf('Minimum grid spacing required: dx_min = %.15g m \n',min_dx)
    fprintf('Minimum # of cells  required: %.2f cells \n',min_cells)
    
    varname = string(vars{1}(2:N_vars+1));
    varname(1) = 'U';
    varname(2) = 'T';

     
% Interpolation
    xStart = 1e-5;
    xEnd = 0.19999;
    OF_gridSize = 10000;
    OF_grid = linspace(xStart,xEnd,OF_gridSize)';
    
    for i = 1:N_vars
        F = griddedInterpolant(x_Cantera,Cantera{i+1},'nearest','nearest') ;
        data = F(OF_grid);
        OF_data{i} = data;
    end
    



% Write data back into OF format
str = sprintf('~/anaconda3/PROJECTS/premixedFlames/%g',tFolder);
cd(str);

for i = 1:N_vars
    fid = fopen(varname(i),'w'); % write permission
    fprintf(fid,'/*--------------------------------*- C++ -*----------------------------------*\\\n');
    fprintf(fid,'| =========                 |                                                 |\n');
    fprintf(fid,'| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n');
    fprintf(fid,'|  \\\\    /   O peration     | Version:  5.x                                   |\n');
    fprintf(fid,'|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n');
    fprintf(fid,'|    \\\\/     M anipulation  |                                                 |\n');
    fprintf(fid,'\\*---------------------------------------------------------------------------*/\n');
    fprintf(fid,'FoamFile\n');
    fprintf(fid,'{\n');
    fprintf(fid,'    version     2.0;\n');
    fprintf(fid,'    format      ascii;\n');
    if strcmp(varname(i),'U')
        fprintf(fid,'    class       volVectorField;\n');
    else
        fprintf(fid,'    class       volScalarField;\n');
    end
    fprintf(fid,'    location    "%.15g";\n',tFolder);
    fprintf(fid,'    object       %s;\n',varname(i));
    fprintf(fid,'}\n');
    fprintf(fid,'// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n');
    fprintf(fid,'\n');
    
    if strcmp(varname(i),'T')
        fprintf(fid,'dimensions      [0 0 0 1 0 0 0];\n');
        fprintf(fid,'\n');
        fprintf(fid,'internalField   nonuniform List<scalar>\n');
        fprintf(fid,'%d\n',OF_gridSize);
        fprintf(fid,'(\n');
        fprintf(fid,'%.15f\n',OF_data{i});
        fprintf(fid,')\n;\n');
        
    elseif strcmp(varname(i),'U')
        fprintf(fid,'dimensions      [0 1 -1 0 0 0 0];\n');
        fprintf(fid,'\n');
        fprintf(fid,'internalField   nonuniform List<vector>\n');
        fprintf(fid,'%d\n',OF_gridSize);
        fprintf(fid,'(\n');
        fprintf(fid,'(%.15f 0.0 0.0)\n',OF_data{i});
        fprintf(fid,')\n;\n');
        
    else
        fprintf(fid,'dimensions      [0 0 0 0 0 0 0];\n');
        fprintf(fid,'\n');
        fprintf(fid,'internalField   nonuniform List<scalar>\n');
        fprintf(fid,'%d\n',OF_gridSize);
        fprintf(fid,'(\n');
        fprintf(fid,'%.15e\n',OF_data{i});
        fprintf(fid,')\n;\n');
    end
    
    
    fprintf(fid,'\nboundaryField\n');
    
    fprintf(fid,'{\n');
    fprintf(fid,'    inlet\n');
    fprintf(fid,'    {\n');
    if strcmp(varname(i),'T')
        fprintf(fid,'        type            fixedValue;\n');
        fprintf(fid,'        value           uniform 300;\n');
    elseif strcmp(varname(i),'C2H4')
        fprintf(fid,'        type            fixedValue;\n');
        fprintf(fid,'        value           uniform 0.0655;\n');
    elseif strcmp(varname(i),'O2')
        fprintf(fid,'        type            fixedValue;\n');
        fprintf(fid,'        value           uniform 0.1963;\n');
    elseif strcmp(varname(i),'N2')
        fprintf(fid,'        type            fixedValue;\n');
        fprintf(fid,'        value           uniform 0.7382;\n');
    elseif strcmp(varname(i),'U')
        fprintf(fid,'        type            fixedValue;\n');
        fprintf(fid,'        value           uniform ( 0.684668 0.000000 0.000000 );\n'); % laminar flame speed from Cantera (f.u[0])
    else 
        fprintf(fid,'        type            fixedValue;\n');
        fprintf(fid,'        value           uniform 0;\n');
    end
    fprintf(fid,'    }\n');
    
    
    fprintf(fid,'    outlet\n');
    fprintf(fid,'    {\n');
        fprintf(fid,'        type            zeroGradient;\n');
    fprintf(fid,'    }\n');
    
    
    fprintf(fid,'    wall\n');
    fprintf(fid,'    {\n');
        fprintf(fid,'        type            empty;\n');
    fprintf(fid,'    }\n');
    
    
    fprintf(fid,'}\n');
    fprintf(fid,'\n');
    fprintf(fid,'\n');
    fprintf(fid,'// ************************************************************************* //\n');
    
    fclose(fid);  
end


