
%% Script to take data from MATLAB and convert to OpenFOAM
%% Xinyu Zhao 2017, Joseph Squeo 2019


function Matlab2OF(tFolder)
    
    fname = 'dataOutput.csv';   % Specify the file name to be opened 
    fid = fopen(fname); % Open file in read mode
    vars = textscan(fid,' %s ','Delimiter',',','MultipleDelimsAsOne',1);
    fclose(fid);        % Close the opened file
    
  
    fid = fopen(fname); % Open file in read mode
    OF = textscan(fid,'%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f','Delimiter',',','headerlines',1,'EmptyValue',0);
    fclose(fid);        % Close the opened file
    N_data = length(OF{1});
    N_vars = length(OF);
    varname = string(vars{1}(1:N_vars));

%READ EXCEL DATA  (exp data is mole fractions for gas species)
%     data = xlsread('dataOutput.xls','A2:AG51081');
%     data(isnan(data)) = 0;
%     N_data = length(data);

% Write data back into OF format
str = sprintf('~/anaconda3/PROJECTS/OpenFOAM_igniteFlame/%g',tFolder);
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
    fprintf(fid,'    class       volScalarField;\n');
    fprintf(fid,'    location    "%.15g";\n',tFolder);
    fprintf(fid,'    object       %s;\n',varname(i));
    fprintf(fid,'}\n');
    fprintf(fid,'// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n');
    fprintf(fid,'\n');
    if strcmp(varname(i),'T')
        fprintf(fid,'dimensions      [0 0 0 1 0 0 0];\n');
        fprintf(fid,'\n');
        fprintf(fid,'internalField   nonuniform List<scalar>\n');
        fprintf(fid,'%d\n',N_data);
        fprintf(fid,'(\n');
        fprintf(fid,'%.2f\n',OF{i});
        fprintf(fid,')\n;\n');
    else
        fprintf(fid,'dimensions      [0 0 0 0 0 0 0];\n');
        fprintf(fid,'\n');
        fprintf(fid,'internalField   nonuniform List<scalar>\n');
        fprintf(fid,'%d\n',N_data);
        fprintf(fid,'(\n');
        fprintf(fid,'%.9e\n',OF{i});
        fprintf(fid,')\n;\n');
    end
    
    
    fprintf(fid,'\nboundaryField\n');
    fprintf(fid,'{\n');
    fprintf(fid,'    nozzle\n');
    fprintf(fid,'    {\n');
    if strcmp(varname(i),'T')
        fprintf(fid,'        type            fixedValue;\n');
        fprintf(fid,'        value           uniform 400;\n');
    elseif strcmp(varname(i),'C2H4')
        fprintf(fid,'        type            fixedValue;\n');
        fprintf(fid,'        value           uniform 1;\n');
    else 
        fprintf(fid,'        type            fixedValue;\n');
        fprintf(fid,'        value           uniform 0;\n');
    end
    fprintf(fid,'    }\n');
    
    fprintf(fid,'    coflow\n');
    fprintf(fid,'    {\n');
    if strcmp(varname(i),'T')
        fprintf(fid,'        type            fixedValue;\n');
        fprintf(fid,'        value           uniform 300;\n');
    elseif strcmp(varname(i),'N2')
        fprintf(fid,'        type            fixedValue;\n');
        fprintf(fid,'        value           uniform 0.767;\n');
    elseif strcmp(varname(i),'O2')
        fprintf(fid,'        type            fixedValue;\n');
        fprintf(fid,'        value           uniform 0.233;\n');
    else 
        fprintf(fid,'        type            fixedValue;\n');
        fprintf(fid,'        value           uniform 0;\n');
    end
    fprintf(fid,'    }\n');
    
    fprintf(fid,'    ambient\n');
    fprintf(fid,'    {\n');
    if strcmp(varname(i),'T')
        fprintf(fid,'        type            fixedValue;\n');
        fprintf(fid,'        value           uniform 300;\n');
    elseif strcmp(varname(i),'N2')
        fprintf(fid,'        type            fixedValue;\n');
        fprintf(fid,'        value           uniform 0.767;\n');
    elseif strcmp(varname(i),'O2')
        fprintf(fid,'        type            fixedValue;\n');
        fprintf(fid,'        value           uniform 0.233;\n');
    else 
        fprintf(fid,'        type            fixedValue;\n');
        fprintf(fid,'        value           uniform 0;\n');
    end
    fprintf(fid,'    }\n');
    
    fprintf(fid,'    outlet\n');
    fprintf(fid,'    {\n');
        fprintf(fid,'        type            zeroGradient;\n');
    fprintf(fid,'    }\n');
    
    fprintf(fid,'    centerside\n');
    fprintf(fid,'    {\n');
        fprintf(fid,'        type            symmetryPlane;\n');
    fprintf(fid,'    }\n');
    
    fprintf(fid,'    outerside\n');
    fprintf(fid,'    {\n');
        fprintf(fid,'        type            zeroGradient;\n');
    fprintf(fid,'    }\n');
    
    fprintf(fid,'    sideplane1\n');
    fprintf(fid,'    {\n');
        fprintf(fid,'        type            symmetryPlane;\n');
    fprintf(fid,'    }\n');
    
    fprintf(fid,'    sideplane2\n');
    fprintf(fid,'    {\n');
        fprintf(fid,'        type            symmetryPlane;\n');
    fprintf(fid,'    }\n');
    
    fprintf(fid,'    burner_wall_1\n');
    fprintf(fid,'    {\n');
    if strcmp(varname(i),'T')
        fprintf(fid,'        type            fixedValue;\n');
        fprintf(fid,'        value           uniform 300;\n');
    else
        fprintf(fid,'        type            zeroGradient;\n');
    end
    fprintf(fid,'    }\n');
    
    fprintf(fid,'    burner_wall_2\n');
    fprintf(fid,'    {\n');
    if strcmp(varname(i),'T')
        fprintf(fid,'        type            fixedValue;\n');
        fprintf(fid,'        value           uniform 300;\n');
    else
        fprintf(fid,'        type            zeroGradient;\n');
    end
    fprintf(fid,'    }\n');
    
    fprintf(fid,'    burner_wall_3\n');
    fprintf(fid,'    {\n');
    if strcmp(varname(i),'T')
        fprintf(fid,'        type            fixedValue;\n');
        fprintf(fid,'        value           uniform 300;\n');
    else
        fprintf(fid,'        type            zeroGradient;\n');
    end
    fprintf(fid,'    }\n');
    
    fprintf(fid,'    burner_wall_4\n');
    fprintf(fid,'    {\n');
    if strcmp(varname(i),'T')
        fprintf(fid,'        type            fixedValue;\n');
        fprintf(fid,'        value           uniform 300;\n');
    else
        fprintf(fid,'        type            zeroGradient;\n');
    end
    fprintf(fid,'    }\n');
    
    fprintf(fid,'}\n');
    fprintf(fid,'\n');
    fprintf(fid,'\n');
    fprintf(fid,'// ************************************************************************* //\n');
    
    fclose(fid);  
end
