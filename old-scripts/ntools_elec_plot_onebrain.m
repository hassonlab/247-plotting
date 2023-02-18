function ntools_elec_plot_onebrain(varargin)

%fprintf('\n Currently editing to read coordinate file format differently - Bobbi')
%fprintf('\n Currently scaling to GloVe scale - Bobbi')

% Plot ECoG electrodes from multiple patients on one brain image.
%% Required inputs: 
% 1. txt file with electrode names, XYZ MNI coordinates, effect numbers (p-values,etc.)
    % -OR- txt file with electrode names, XYZ MNI coordinates only
% 2. surface plot of average brain: 'ch2_template_mni_lh_pial.mat' for one
    % hemisphere; {'ch2_template_mni_lh_pial.mat','ch2_template_mni_rh_pial.mat'} 
    % for both hemispheres
% 3. 'effect'/'colors': idicate whether the text files contains an effect
    % to color map a 1x3 color vertices
%% Optional inputs:
% To use optional inputs indidcate the variable name followed by the value
% you want to assign to it.
% (Additional to the ntools_elec_plot optional inputs)
% * Plot title: ...,'plot_title','Podcast'); default = no title
% * Colorbar title: ...,'cbar_title','p-value'); default = no title
% * Colorbar tick labels ...,'tick_labels',{'0.05','0.04','0.03','0.02','0.01','0'})
% * outSpecifier: ...,'outSpecifier','_Prediction' --> If you want to add additional
% description to the default output file name

%%
% a stand-alone program that shows ieeg electrodes on the pial surface and
% save the images into textfile folder/images/. Default saving name is
% PatientID(NYxx)_space_elecLabel_viewpoint_hemisphere.png
% 
% space could be T1 or MNI
% elecLabel could be grid, strip, depth, or grid and strip
% viewpoint could be left, right, top, below, back or front
% hemisphere could be lh, rh, or both
% 
% required input:
% elec_text: text file with xyz electrode coordinates
% pial_mat: matlab structure with pial surface
%
% optional input:
% plot: part to plot 
%       'G': Grid only 
%       'S': Strip only 
%       'D': Depth Only 
%       'GS': Both Grid and Strip  
%       'brain': brain image only
% showlabel: to show the electrodes' labels (1) or not (0)
% saveimg: to save the images (1) or not (0),if set to (10), save in silence
% aparc: plot the image with aparc.annot, all electrodes and labels will be white
%
% Usage: run ntools_elec_plot in command window
% the gui is prompted out for file selection
% or: ntools_elec_plot(fname, {'lh.mat', 'rh.mat'})
%     ntools_elec_plot(fname, l(r)h.mat);
%     ntools_elec_plot(fname, l(r)h.mat,'showlabel',1,'saveimg',10);
%     ntools_elec_plot(fname,lh.mat,'plot','G','aparc','lh.aparc.annot');
%
% also see: http://ieeg.pbworks.com/Viewing-Electrode-Locations
%
% written by  Hugh Wang, Xiuyuan.Wang@nyumc.org, May 13, 2009
%
% modified on May 14, 2009 by Hugh
% make judgement on the input file type and not sensitive to the order of 
% input variable. add the option to show the electrodes' labels or not.
%
% modified on July 22nd, 2009 by Hugh
% for subjects who has electrodes on both hemisphere, loading the both
% pial.mat file will generate the image with whole brain and save the
% images from 6 views (left, right, top, below, back & front)
%
% modified on Aug 8th, 2009 by Hugh
% show only lh(rh)'s electrodes if choose lh(rh)_surf.mat
%
% modified on Jan 28th, 2010 by Hugh
% default saving name
%
% modified on Jan 13th, 2014 by Hugh
% more optional inputs 'plot' and 'aparc'
% 
% modified on Feb 11th, 2019 by Hugh
% support for experimental grid


%% Get the elec info
if nargin==0
    [FileName,PathName] = uigetfile('*.txt','Select the electrodes text file',pwd'); 
    [surfname, surfpath] = uigetfile('*.mat','Select the patient brain surf',PathName,'MultiSelect','on');
    surf = strcat(surfpath,surfname);      
elseif nargin>=2
    aa = strfind(varargin{1},'/');
    if ~isempty(aa)
        FileName = varargin{1}(aa(end)+1:end);
        PathName = varargin{1}(1:aa(end));
    else
        FileName = varargin{1};
        PathName = [pwd,'/'];
    end
    surf = varargin{2}; 

    try labelshow = varargin{find(strcmp('showlabel',varargin))+1}; catch err; end
    try genimg = varargin{find(strcmp('saveimg',varargin))+1}; catch err; end
    try plt = varargin{find(strcmp('plot',varargin))+1}; catch err; end
    try aparc = varargin{find(strcmp('aparc',varargin))+1}; catch err; end
    try plot_title = varargin{find(strcmp('plot_title',varargin))+1}; catch err; end
    try cbar_title = varargin{find(strcmp('cbar_title',varargin))+1}; catch err; end
    try tick_labels = varargin{find(strcmp('tick_labels',varargin))+1}; catch err; end
    try outSpecifier = varargin{find(strcmp('outSpecifier',varargin))+1}; catch err; end
    
    effect = varargin{3};
%    FileName2 = varargin{4};

end

if strcmp(effect,'effect')
    if exist(fullfile(PathName, FileName),'file')
        if strcmp(FileName(end-3:end),'.csv')
            elec_all = readtable(fullfile(PathName, FileName));
            %elec_all_temp = readtable(fullfile(PathName, 'podcast_MNIcoor_gloveSigElec_maxCorr.csv'));
            %elec_all = elec_all(elec_all.glove_sigElec == 1,:);
        else
            fid = fopen(fullfile(PathName, FileName));
            %fidTemp = fopen(fullfile(PathName, FileName2));
            %elec_all = textscan(fid,'%s %f %f %f %s %f');
            %elec_all_temp = textscan(fidTemp,'%s %f %f %f %s %f');
            fclose(fid);
            elec_all = readtable(fullfile(PathName, FileName));
        end
    end
%             effect = elec_all.Var6;
%             correlation_max = 0.1;
%             correlation_min = -0.1;
%             effect1 = effect;
%             effect2 = effect;
%             effect1(effect1 < 0) = 0;
%             effect2(effect2 > 0) = 0;
% %            adj_effect = effect -  correlation_min; %0.0689;
%             col_dif = 1 / correlation_max; %2.4009;
%             col_dif2 = 1 / correlation_min;
%             col_mark = 1 - effect1  * col_dif;
%             col_mark2 = 1 - effect2 * col_dif2;
%             %col_mark = ones(height(elec_all),1); % TEMP\
%             colors = [col_mark2 ones(size(elec_all,1),1) col_mark];
%             disp(colors);

            effect_max = 0.3;
            effect_min = 0.18;
            elec_all = elec_all(elec_all.Var6 >= effect_min,:);
            effect = elec_all.Var6;
            
            effect(effect >= effect_max) = effect_max;
            adj_effect1 = effect - effect_min;
%             adj_effect1 = effect;
            col_dif = 1 / (effect_max - effect_min);
            col_mark = adj_effect1 * col_dif;


            colors = [ones(size(col_mark,1),1) col_mark zeros(size(col_mark,1),1)];

%             toohigh = colors(:,2) > 1;
%             colors(toohigh,2) = 1;
% 
%             insig = colors(:,2) < 0;
%             colors(insig,:) = 0;


elseif strcmp(effect,'colors')
    if strcmp(FileName(end-3:end),'.csv')
        elec_all = readtable(fullfile(PathName, FileName));
        %elec_all = elec_all(elec_all.glove_sigElec == 1,:);
        %elec_all = elec_all(table2array(elec_all(:,8)) == 0.5010,:);
%         if exist(fullfile(PathName, FileName),'file')
%             fid = fopen(fullfile(PathName, FileName));
%             elec_all = textscan(fid,'%s %f %f %f %s %f %f %f');
%             fclose(fid);
%         end
    else
        elec_all = readtable(fullfile(PathName, FileName));
    end
        colors = [elec_all.Var6 elec_all.Var7 elec_all.Var8];

elseif strcmp(effect,'none')
    %if strcmp(FileName(end-3:end),'.csv')
        elec_all = readtable(fullfile(PathName, FileName));
        %elec_all = elec_all(~strcmp(elec_all.Var6,'EG'),:);
        %elec_all = elec_all(elec_all.Var1 == 742,:);
%         elec_all(elec_all.Var1 == 662,:) = [];
%         elec_all(elec_all.Var1 == 723,:) = [];
%         elec_all(elec_all.Var1 == 741,:) = [];
%         elec_all(elec_all.Var1 == 743,:) = [];
%         elec_all(elec_all.Var1 == 763,:) = [];
        % subj = table2array(elec_all(:,1));

        elec_all = elec_all(:,1:5);
        colors = [zeros(size(elec_all,1),1) zeros(size(elec_all,1),1) zeros(size(elec_all,1),1)];
%         E = readtable(fullfile(PathName, 'NY742_autoNY742_coor_T1_2019-08-07.txt'));
%         for X = 1:height(elec_all)
%             if length(elec_all.Var2{X}) == 2 && ~strcmp(elec_all.Var6(X),'S')
%                 lbl = elec_all.Var2{X};
%                 elec_all.Var2{X} = [lbl(1) '00' lbl(2)];
%             elseif length(elec_all.Var2{X}) == 3 && ~strcmp(elec_all.Var6(X),'S')
%                 lbl = elec_all.Var2{X};
%                 elec_all.Var2{X} = [lbl(1) '0' lbl(2:3)];
%             elseif length(elec_all.Var2{X}) == 3 && strcmp(elec_all.Var6(X),'S')
%                 lbl = elec_all.Var2{X};
%                 elec_all.Var2{X} = [lbl(1:2) '0' lbl(3)];
%             end
%             f = find(strcmp(elec_all.Var2(X),E.Var1));
%             elec_all.Var3(X) = E.Var2(f);
%             elec_all.Var4(X) = E.Var3(f);
%             elec_all.Var5(X) = E.Var4(f);
%         end
        %elec_all = elec_all(strcmp(elec_all.Region,'cSTG'),:);
    %end
%         for numsubj = 1:size(colors,1)
%             if subj(numsubj) == 717
%                 colors(numsubj,:) = [0 0.4470 0.7410];
%             elseif subj(numsubj) == 742
%                 colors(numsubj,:) = [0.4660 0.6740 0.1880];
%             elseif subj(numsubj) == 798
%                 colors(numsubj,:) = [0.6350 0.0780 0.1840];
%             end
%         end
end

%if strcmp(FileName(end-3:end),'.csv')
% elec_all.Properties.VariableNames = {'electrodeName','X','Y','Z','type','r','g','b'};
elec_all.Properties.VariableNames = {'electrodeName','X','Y','Z','type','effect'};
    elec_cell = [cellstr(elec_all.electrodeName),num2cell(elec_all.X),num2cell(elec_all.Y),...
        num2cell(elec_all.Z),num2cell(colors)];
% else
%     elec_cell = [elec_all{1},num2cell(elec_all{2}),num2cell(elec_all{3}),num2cell(elec_all{4}),num2cell(colors)];
% end

% else
% %     elec_cell = [];
%     disp('No electrode was found. Please check you input text file.')
%     return
%end

%% Get the filename info
b = strfind(FileName,'_');
Pname = FileName(1:b(1)-1);

if contains(upper(FileName),'T1')
    space = '_T1_';
elseif contains(upper(FileName),'MNI')
    space = '_MNI_';
else
    space = '_';
end

if length(surf)==2
    sph = 'both';
else
    sph = regexpi(surf,'[r,l]h','match');
    sph = char(sph{:});
end

%% Separate Grid, Strip and Depth electrodes

g = strncmpi('G',elec_all.type,1);
d = strncmpi('D',elec_all.type,1);
eg = strncmpi('EG',elec_all.type,2);

elec_grid = elec_cell(g,:);
elec_expGrid = elec_cell(eg,:);
elec_depth = elec_cell(d,:);
elec_cell(logical(g+d+eg),:) = [];


%% Plot the elecs
% what to plot
if ~exist('plt','var')
    plt = menu('What part do you want to plot?','Grid only', 'Strip only','Depth Only',...
        'Both Grid and Strip','Brain only');
end
% show label
if ~exist('labelshow','var')
    labelshow = menu('Do you want to show the label?','Yes','No');
end

% plot with aparc
if ~exist('aparc','var')
    aparc = menu('Do you want to plot with aseg parcellations? (note: all electrodes and labels will be white)',...
        'Yes','No');
    if aparc==1 
%         aparc = uigetdir(getenv('SUBJECTS_DIR'),'Select subject FS recon folder');
        [aparc,aparc_path]= uigetfile('*.annot','Select the aparc annotation file',PathName,'MultiSelect','on');
        aparc = fullfile(aparc_path,aparc);
    else
        aparc = [];
    end
elseif isempty(aparc) || isnumeric(aparc)
    aparc = [];
end

% save image
if ~exist('genimg','var')
    genimg = menu('Do you want to save the images?','Yes', 'No');
end

% load pial surfs and corresponding aparc annotation file
if strcmp(sph,'both')
    surf_brain.sph1 = load(surf{1});
    surf_brain.sph2 = load(surf{2});

    if ~isempty(aparc)
        hemi1 = regexpi(surf{1},'[r,l]h','match'); hemi1 = char(hemi1{:});
%         aparc_annot.hemi1 = char(aparc(strncmpi(hemi1,aparc,2)));
        idx = strfind(aparc,hemi1);
        idx(cellfun(@isempty,idx)) = {0};
        aparc_annot.hemi1 = char(aparc(logical(cell2mat(idx))));
        
        hemi2 = regexpi(surf{2},'[r,l]h','match'); hemi2 = char(hemi2{:});
%         aparc_annot.hemi2 = char(aparc(strncmpi(hemi2,aparc,2)));
        idx = strfind(aparc,hemi2);
        idx(cellfun(@isempty,idx)) = {0};
        aparc_annot.hemi2 = char(aparc(logical(cell2mat(idx))));
    else
        aparc_annot = [];
    end
    
else 
    surf_brain = load(surf);
%     if ~isempty(aparc), aparc_annot = strcat(aparc,'/label/',sph,'.aparc.annot'); else aparc_annot = []; end
    aparc_annot = aparc;
end

% main plot
if (isequal(plt,1) || strcmpi(plt,'G')) && ~isempty(elec_grid)
    showpart = 'G';
    nyu_plot(surf_brain,sph,cell2mat(elec_grid(:,2:4)),char(elec_grid(:,1)),...
        cell2mat(elec_grid(:,5:7)),labelshow,aparc_annot);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% experimental Grid
    if ~isempty(elec_expGrid)
        hold on;
        expGrid_radius = 2;
        
        for i=1:size(elec_expGrid,1)
            if isempty(aparc_annot)
                expGrid_color = cell2mat(elec_expGrid(i,5:7));
            else % re-define strip color if plot with aparc
                expGrid_color = cell2mat(elec_expGrid(i,5:7));
            end            
            
            plotSpheres(elec_expGrid{i,2},elec_expGrid{i,3},elec_expGrid{i,4},...
                expGrid_radius,expGrid_color);
            if labelshow==1
                [xx, yy, zz] = adjust_elec_label([elec_expGrid{i,2:4}],expGrid_radius); % default radius = 2
                text('Position',[xx yy zz],'String',elec_expGrid(i,1),'Color','w',...
                    'VerticalAlignment','top');
            end    
        end
        hold off;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif (isequal(plt,2) || strcmpi(plt,'S')) && ~isempty(elec_cell)
    showpart = 'S';
    nyu_plot(surf_brain,sph,cell2mat(elec_cell(:,2:4)),char(elec_cell(:,1)),...
        cell2mat(elec_cell(:,5:7)),labelshow,aparc_annot);
elseif (isequal(plt,3) || strcmpi(plt,'D')) && ~isempty(elec_depth)
    showpart = 'D';
    nyu_plot(surf_brain,sph,cell2mat(elec_depth(:,2:4)),char(elec_depth(:,1)),...
        cell2mat(elec_depth(:,5:7)),labelshow,[],1.5,0.3);
elseif (isequal(plt,4) || strcmpi(plt,'GS')) %&& ~isempty(elec_grid) && ~isempty(elec_cell)
    showpart = 'GS';
    elec = cell2mat(elec_cell(:,2:4));
    elec_name = char(elec_cell(:,1));
    nyu_plot(surf_brain,sph,cell2mat(elec_grid(:,2:4)),char(elec_grid(:,1)),...
        cell2mat(elec_grid(:,5:7)),labelshow,aparc_annot); hold on;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% experimental Grid
    if ~isempty(elec_expGrid)
        hold on;
        expGrid_radius = 2;
        
        for i=1:size(elec_expGrid,1)
            if isempty(aparc_annot)
                expGrid_color = cell2mat(elec_expGrid(i,5:7));
            else % re-define strip color if plot with aparc
                expGrid_color = cell2mat(elec_expGrid(i,5:7));
            end
            
            plotSpheres(elec_expGrid{i,2},elec_expGrid{i,3},elec_expGrid{i,4},...
                expGrid_radius,expGrid_color);
            if labelshow==1
                [xx, yy, zz] = adjust_elec_label([elec_expGrid{i,2:4}],expGrid_radius); % default radius = 2
                text('Position',[xx yy zz],'String',elec_expGrid(i,1),'Color','w','VerticalAlignment','top');
            end    
        end
        hold off;        
    end    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    for i=1:size(elec,1)
        if isempty(aparc_annot)
            stripcolor = cell2mat(elec_cell(i,5:7));
        else % re-define strip color if plot with aparc
            stripcolor = cell2mat(elec_cell(i,5:7));
        end
        plotSpheres(elec(i,1),elec(i,2),elec(i,3),2,stripcolor);
        if labelshow==1
            [xx, yy, zz] = adjust_elec_label(elec(i,:)); % default radius = 2
            text('Position',[xx yy zz],'String',elec_name(i,:),'Color','k','VerticalAlignment','top');
        end
    end
    hold off;  
elseif isequal(plt,5) || strcmpi(plt,'brain')
    showpart = 'Brain';
    nyu_plot(surf_brain,[],[],[]);
else
    disp('sorry, the electrodes you chose to plot are not on the surface you loaded');
    return;
end

%% save images

%% Key
% a = ' Participant 3';
% b = ' Participant 2';
% c = ' Participant 1';
% d = ' Participant 6';
% e = ' Participant 5';
% f = ' Participant 4';
% g = ' Participant 3';
% h = ' Participant 2';
% i = ' Participant 1';
% plotSpheres(-100,-100,40,2,[0.6350 0.0780 0.1840]);
%     %[xx, yy, zz] = adjust_elec_label(a); % default radius = 2
%     text('Position',[-102,-102,42.5],'String',a,'Color', 'k','VerticalAlignment','top');
% plotSpheres(-100,-100,45,2,[0.4660 0.6740 0.1880]);
%     %[xx, yy, zz] = adjust_elec_label(b); % default radius = 2
%     text('Position',[-102,-102,47.5],'String',b,'Color','k','VerticalAlignment','top');
% plotSpheres(-100,-100,50,2,[0 0.4470 0.7410]);
%     %[xx, yy, zz] = adjust_elec_label(b); % default radius = 2
%     text('Position',[-102,-102,52.5],'String',c,'Color','k','VerticalAlignment','top');
% plotSpheres(-100,-100,55,2,[0.466 0.674 0.188]);
%     %[xx, yy, zz] = adjust_elec_label(b); % default radius = 2
%     text('Position',[-102,-102,57.5],'String',d,'Color','k','VerticalAlignment','top');
% plotSpheres(-100,-100,60,2,[0.301 0.745 0.933]);
%     %[xx, yy, zz] = adjust_elec_label(b); % default radius = 2
%     text('Position',[-102,-102,62.5],'String',e,'Color','k','VerticalAlignment','top');
% plotSpheres(-100,-100,65,2,[0.635 0.078 0.184]);
%     %[xx, yy, zz] = adjust_elec_label(b); % default radius = 2
%     text('Position',[-102,-102,67.5],'String',f,'Color','k','VerticalAlignment','top');
% plotSpheres(-100,-100,70,2,[0 0.4470 0.7410]);
%     %[xx, yy, zz] = adjust_elec_label(b); % default radius = 2
%     text('Position',[-102,-102,72.5],'String',g,'Color','k','VerticalAlignment','top');
% plotSpheres(-100,-100,75,2,[0.8500 0.3250 0.0980]);
%     %[xx, yy, zz] = adjust_elec_label(b); % default radius = 2
%     text('Position',[-102,-102,77.5],'String',h,'Color','k','VerticalAlignment','top');
% plotSpheres(-100,-100,80,2,[0.9290 0.6940 0.1250]);
%     %[xx, yy, zz] = adjust_elec_label(b); % default radius = 2
%     text('Position',[-102,-102,82.5],'String',i,'Color','k','VerticalAlignment','top');
%%

if exist('plot_title','var')
    title(plot_title)
end

if exist('cbar_title','var')
    map_colors = flip(0:0.02:1);
    map = [ones(length(map_colors),1) (1 - map_colors') zeros(length(map_colors),1)];
    colormap(map);
%all_colors(51,:)
%        colormap(all_colors);
    caxis([0 1])
    if exist('tick_labels','var')
        c = colorbar('Ticks',0:0.125:1,...
        'TickLabels',tick_labels);
    else
        %elec_all_temp = elec_all.effect;
        maxMinDiff = (ceil(effect_max*100)/100) - (floor(effect_min*100)/100);
        stepSize = maxMinDiff/4;
        ticks = (floor(effect_min*100)/100):stepSize:(ceil(effect_max*100)/100);

        ticks_new = num2str(ticks');
        ticks_new_2 = cellstr(ticks_new);
        c = colorbar('Ticks',0:0.25:1,'TickLabels',ticks_new_2');
    end
    c.FontSize = 16;
    %c.Location = 'northoutside';
    title(c,cbar_title)
end

if genimg==1 || genimg==10
    if ~exist([PathName 'images/'],'dir')
        mkdir([PathName 'images/']);
    end
    
    if labelshow==1
        label = '_label';
    else
        label = [];
    end
    
    if ~isempty(aparc)
        % split aparc file name and get the middle one in output image file
        % name
        aparc_fnames = regexp(aparc,'\.','split');
%        aparc = ['_',aparc_fnames{3}];
    end
    
    if ~exist('outSpecifier','var')
        outSpecifier = '';
    end
        
    format = 'png';
    outimgfname = [PathName,'images/',Pname,'_XaxisviewsX_',sph,...
        label,outSpecifier];
    
    % save the images in silence
    if genimg==10, set(gcf,'Visible','off'); end
    
    f = gcf;
    
    if strcmp(sph,'lh')
        view(270, 0);
        saveas(gcf,regexprep(outimgfname,'XaxisviewsX','lateral'),format);
%         view(90,0);
%         saveas(gcf,regexprep(outimgfname,'XaxisviewsX','mesial'),format);
        
    elseif strcmp(sph,'rh')
%         view(270, 0);
%         saveas(gcf,regexprep(outimgfname,'XaxisviewsX','mesial'),format);
        view(90,0);
        saveas(gcf,regexprep(outimgfname,'XaxisviewsX','lateral'),format);
        
    elseif strcmp(sph,'both')
        view(270, 0);
        saveas(gcf,regexprep(outimgfname,'XaxisviewsX','left'),format);
        view(90,0);
        saveas(gcf,regexprep(outimgfname,'XaxisviewsX','right'),format);
    end
%     view(0,0);
%     saveas(gcf,regexprep(outimgfname,'XaxisviewsX','posterior'),format);
% 
%     view(180,0);
%     saveas(gcf,regexprep(outimgfname,'XaxisviewsX','frontal'),format);
% 
%     view(90,90);
%     saveas(gcf,regexprep(outimgfname,'XaxisviewsX','dorsal'),format);
% 
%     view(90,-90);
%     set(light,'Position',[1 0 -1]);
%     saveas(gcf,regexprep(outimgfname,'XaxisviewsX','ventral'),format);
else 
    return;
end

end

%% subfunctions 
%% nyu_plot
function nyu_plot(surf_brain,sph,elec,elecname,color,label,aparc_annot,radius,alpha)

if ~exist('color','var')
    color = 'w';
end
if ~exist('label','var')
    label = 0;
end
if ~exist('alpha','var')
    alpha = 1;
end
if ~exist('radius','var')
    radius = 2;
end
if ~exist('aparc_annot','var')
    aparc_annot = [];
end

figure;

col = [.7 .7 .7];

if strcmp(sph,'both')
    sub_sph1.vert = surf_brain.sph1.coords;
    sub_sph1.tri = surf_brain.sph1.faces;

    sub_sph2.vert = surf_brain.sph2.coords;
    sub_sph2.tri = surf_brain.sph2.faces;
    
    if isempty(aparc_annot) || ~exist(aparc_annot.hemi1,'file') || ~exist(aparc_annot.hemi2,'file')
        col1=repmat(col(:)', [size(sub_sph1.vert, 1) 1]);
        col2=repmat(col(:)', [size(sub_sph2.vert, 1) 1]);
    else
        [~,albl1,actbl1]=fs_read_annotation(aparc_annot.hemi1);
        [~,aa] = ismember(albl1,actbl1.table(:,5));
        aa(aa==0) = 1;
        col1 = actbl1.table(aa,1:3)./255;
        
        [~,albl2,actbl2]=fs_read_annotation(aparc_annot.hemi2);
        [~,bb] = ismember(albl2,actbl2.table(:,5));
        bb(bb==0) = 1;
        col2 = actbl2.table(bb,1:3)./255;
                
        % re-define the electrode color if plot with aparc
        %color = 'w';
    end    
    
    trisurf(sub_sph1.tri, sub_sph1.vert(:, 1), sub_sph1.vert(:, 2),sub_sph1.vert(:, 3),...
        'FaceVertexCData', col1,'FaceColor', 'interp','FaceAlpha',alpha);
    hold on;
    trisurf(sub_sph2.tri, sub_sph2.vert(:, 1), sub_sph2.vert(:, 2), sub_sph2.vert(:, 3),...
        'FaceVertexCData', col2,'FaceColor', 'interp','FaceAlpha',alpha);
else    
    if isfield(surf_brain,'coords')==0
        sub.vert = surf_brain.surf_brain.coords;
        sub.tri = surf_brain.surf_brain.faces;
    else
        sub.vert = surf_brain.coords;
        sub.tri = surf_brain.faces;
    end
    
    if isempty(aparc_annot) || ~exist(aparc_annot,'file')
        col=repmat(col(:)', [size(sub.vert, 1) 1]);
    else
        [~,albl,actbl]=fs_read_annotation(aparc_annot);
        [~,cc] = ismember(albl,actbl.table(:,5));
        cc(cc==0) = 1;
        col = actbl.table(cc,1:3)./255;
        
        % re-define the electrode color if plot with aparc
        %color = 'w';
    end    
        
    trisurf(sub.tri, sub.vert(:, 1), sub.vert(:, 2), sub.vert(:, 3),...
        'FaceVertexCData', col,'FaceColor', 'interp','FaceAlpha',alpha);
end

shading interp;
lighting gouraud;
material dull;
light;
axis off
hold on;
for i=1:size(elec,1)
    plotSpheres(elec(i,1),elec(i,2),elec(i,3),radius,color(i,:));
    if label==1
        [x, y, z] = adjust_elec_label(elec(i,:),radius);
        text('Position',[x y z],'String',elecname(i,:),'Color','w','VerticalAlignment','top');
    end
end
set(light,'Position',[-1 0 1]); 
    if strcmp(sph,'lh')
        view(270, 0);      
    elseif strcmp(sph,'rh')
        view(90,0);        
    elseif strcmp(sph,'both')
        view(90,90);
    end
set(gcf, 'color','white','InvertHardCopy', 'off');
axis tight;
axis equal;
end

%% adjust_elec_label
function [x, y, z] = adjust_elec_label(elec,radius)

if ~exist('radius','var')
    radius = 2;
end

if elec(1)>0
    x = elec(1)+radius;
else
    x = elec(1)-radius;
end

if elec(3)>0
    z = elec(3)+radius;
else
    z = elec(3)-radius;
end

y = elec(2);

end

%% plotSpheres
function [shand]=plotSpheres(spheresX, spheresY, spheresZ, spheresRadius,varargin)

if iscell(spheresX)
spheresX = cell2mat(spheresX);
spheresX = str2double(spheresX);
spheresY = cell2mat(spheresY);
spheresY = str2double(spheresY);
spheresZ = cell2mat(spheresZ);
spheresZ = str2double(spheresZ);
end

if nargin>4
    col=varargin{:};
end

spheresRadius = ones(length(spheresX),1).*spheresRadius;
% set up unit sphere information
numSphereFaces = 25;
[unitSphereX, unitSphereY, unitSphereZ] = sphere(numSphereFaces);

% set up basic plot
sphereCount = length(spheresRadius);

% for each given sphere, shift the scaled unit sphere by the
% location of the sphere and plot
for i=1:sphereCount
sphereX = spheresX(i) + unitSphereX*spheresRadius(i);
sphereY = spheresY(i) + unitSphereY*spheresRadius(i);
sphereZ = spheresZ(i) + unitSphereZ*spheresRadius(i);
shand=surface(sphereX, sphereY, sphereZ,'FaceColor',col,'EdgeColor','none','AmbientStrength',0.7);
end

end
