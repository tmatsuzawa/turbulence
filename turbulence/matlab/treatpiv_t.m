%% PIV caller

function treatpiv_t(Dirbase,dirlist,Dt,W,step,test)

subratio = 10;
Data_name = ['/PIV_W' num2str(W) '_step' num2str(step/2) '_data'];

if (test)
    Dir = Dirbase;
else
    Dir = [Dirbase '/Tiff_folder'];
end

if (W==16)
    N=4;
elseif (W==32)
    N=3;
elseif (W==64)
    N=2;
else
    disp('No valid box size given')
end

disp(length(dirlist))

for ind = 1:1:length(dirlist)
    directory=dirlist{ind};
    files= dir([Dir '/' directory '/*.tiff']);
    
    disp('Files index loaded')
    filenames={files.name};
    filenames = sortrows(filenames); %sort all image files
    amount = length(filenames);
    %% Standard PIV Settings
    s = cell(10,2); % To make it more readable, let's create a "settings table"
    %Parameter                       %Setting           %Options
    s{1,1}= 'Int. area 1';           s{1,2}=64;         % window size of first pass
    s{2,1}= 'Step size 1';           s{2,2}=32;         % step of first pass
    s{3,1}= 'Subpix. finder';        s{3,2}=2;          % 1 = 3point Gauss, 2 = 2D Gauss
    s{4,1}= 'Mask';                  s{4,2}=[];         % If needed, generate via: imagesc(image); [temp,Mask{1,1},Mask{1,2}]=roipoly;
    s{5,1}= 'ROI';                   s{5,2}=[];         % Region of interest: [x,y,width,height] in pixels, may be left empty
    s{6,1}= 'Nr. of passes';         s{6,2}=N;          % 1-4 nr. of passes
    s{7,1}= 'Int. area 2';           s{7,2}=32;         % second pass window size
    s{8,1}= 'Int. area 3';           s{8,2}=16;         % third pass window size
    s{9,1}= 'Int. area 4';           s{9,2}=16;         % fourth pass window size
    s{10,1}='Window deformation';    s{10,2}='spline'; % '*spline' is more accurate, but slower
    
    % Standard image preprocessing settings
    p = cell(8,1);
    %Parameter                       %Setting           %Options
    p{1,1}= 'ROI';                   p{1,2}=s{5,2};     % same as in PIV settings
    p{2,1}= 'CLAHE';                 p{2,2}=1;          % 1 = enable CLAHE (contrast enhancement), 0 = disable
    p{3,1}= 'CLAHE size';            p{3,2}=10;         % CLAHE window size
    p{4,1}= 'Highpass';              p{4,2}=1;          % 1 = enable highpass, 0 = disable
    p{5,1}= 'Highpass size';         p{5,2}=15;         % highpass size
    p{6,1}= 'Clipping';              p{6,2}=1;          % 1 = enable clipping, 0 = disable
    p{7,1}= 'Clipping thresh.';      p{7,2}=1;          % 1 = enable wiener noise removing, 0 = disable
    p{8,1}= 'Intensity Capping';     p{8,2}=3;          % 0-255 wiener parameter
    
    
    % PIV analysis loop
    x=cell(1);
    y=x;
    u=x;
    v=x;
    typevector=x; %typevector will be 1 for regular vectors, 0 for masked areas
    typemask=x;
    counter=0;
    
    %create the directory to save the data :
    basename = directory(1:end-5); %remove the _File extension
    PathName =[Dirbase Data_name '/PIVlab_ratio2_W' int2str(s{5+s{6,2},2}) 'pix_Dt_' int2str(Dt) '_' basename];
    mkdir(PathName);

    for i=1:step:(amount-Dt)
        %file_conv = [Dir 'Corr_map_128pix_Dt' num2str(Dt)];
        %write result in a txt file
        ndigit = floor(log10(amount*subratio))+1;
        number = str2num(filenames{i}(3:7));        
        disp(number)
        if number>0
            nzero = ndigit -(floor(log10(number))+1);
        else
            nzero = ndigit -(floor(log10(1))+1)+1;
        end
        
        zeros='';
        for k=1:nzero
            zeros=[zeros '0'];
        end
        FileName = ['D' zeros int2str(number) '.txt'];
        
        %   disp(FileName)
        disp(fullfile(PathName,FileName))
        if exist(fullfile(PathName,FileName))~=2
            %  disp(i+1)
            counter=counter+1;
            image1 = imread(fullfile([Dir '/' directory], filenames{i})); % read images
            image2 = imread(fullfile([Dir '/' directory], filenames{i+Dt}));
            image1 = PIVlab_preproc (image1,p{1,2},p{2,2},p{3,2},p{4,2},p{5,2},p{6,2},p{7,2},p{8,2}); %preprocess images
            image2 = PIVlab_preproc (image2,p{1,2},p{2,2},p{3,2},p{4,2},p{5,2},p{6,2},p{7,2},p{8,2});
            [x{1},y{1},u{1},v{1},typevector{1}] = piv_FFTmulti (image1,image2,s{1,2},s{2,2},s{3,2},s{4,2},s{5,2},s{6,2},s{7,2},s{8,2},s{9,2},s{10,2});%,file_conv);
                                                  %piv_FFTmulti (image1,image2,interrogationarea, step, subpixfinder, mask_inpt, roi_inpt,passes,int2,int3,int4,imdeform)
            
            typemask{1} = logical(not(isnan(u{1}))+not(isnan(v{1})));
            
            clc
            disp(['PIV all fields:' int2str(i/(amount-1)*100) ' %']); % displays the progress in command window
         
            % PIV postprocessing loop
            % Settings
            umin = -2; % minimum allowed u velocity
            umax = 2; % maximum allowed u velocity
            vmin = -2; % minimum allowed v velocity
            vmax = 2; % maximum allowed v velocity
            stdthresh=6; % threshold for standard deviation check
            epsilon=0.15; % epsilon for normalized median test
            thresh=5; % threshold for normalized median test

            u_filt=cell(amount/2,1);
            v_filt=u_filt;
            typevector_filt=u_filt;
            for PIVresult=1:size(x,1)
                u_filtered=u{PIVresult,1};
                v_filtered=v{PIVresult,1};
                typevector_filtered=typevector{PIVresult,1};
                %vellimit check
                u_filtered(u_filtered<umin)=NaN;
                u_filtered(u_filtered>umax)=NaN;
                v_filtered(v_filtered<vmin)=NaN;
                v_filtered(v_filtered>vmax)=NaN;
                % stddev check
                meanu=nanmean(nanmean(u_filtered));
                meanv=nanmean(nanmean(v_filtered));
                std2u=nanstd(reshape(u_filtered,size(u_filtered,1)*size(u_filtered,2),1));
                std2v=nanstd(reshape(v_filtered,size(v_filtered,1)*size(v_filtered,2),1));
                minvalu=meanu-stdthresh*std2u;
                maxvalu=meanu+stdthresh*std2u;
                minvalv=meanv-stdthresh*std2v;
                maxvalv=meanv+stdthresh*std2v;
                u_filtered(u_filtered<minvalu)=NaN;
                u_filtered(u_filtered>maxvalu)=NaN;
                v_filtered(v_filtered<minvalv)=NaN;
                v_filtered(v_filtered>maxvalv)=NaN;
                % normalized median check
                %Westerweel & Scarano (2005): Universal Outlier detection for PIV data
                [J,I]=size(u_filtered);
                medianres=zeros(J,I);
                normfluct=zeros(J,I,2);
                b=1;
                for c=1:2
                    if c==1; velcomp=u_filtered;else;velcomp=v_filtered;end %#ok<*NOSEM>
                    for i=1+b:I-b
                        for j=1+b:J-b
                            neigh=velcomp(j-b:j+b,i-b:i+b);
                            neighcol=neigh(:);
                            neighcol2=[neighcol(1:(2*b+1)*b+b);neighcol((2*b+1)*b+b+2:end)];
                            med=median(neighcol2);
                            fluct=velcomp(j,i)-med;
                            res=neighcol2-med;
                            medianres=median(abs(res));
                            normfluct(j,i,c)=abs(fluct/(medianres+epsilon));
                        end
                    end
                end
                info1=(sqrt(normfluct(:,:,1).^2+normfluct(:,:,2).^2)>thresh);
                u_filtered(info1==1)=NaN;
                v_filtered(info1==1)=NaN;

                typevector_filtered(isnan(u_filtered))=2;
                typevector_filtered(isnan(v_filtered))=2;
                typevector_filtered(typevector{PIVresult,1}==0)=0; %restores typevector for mask

                %Interpolate missing data
                u_filtered=inpaint_nans(u_filtered,4);
                v_filtered=inpaint_nans(v_filtered,4);

                u_filt{PIVresult,1}=u_filtered;
                v_filt{PIVresult,1}=v_filtered;
                typevector_filt{PIVresult,1}=typevector_filtered;
            end
            % save the data in a .txt file
            save_single_data_PIVlab(ff,Dt,PathName,FileName,filenames,false,x{1},y{1},u_filt{1},v_filt{1})
        else
            if (i==1)
                disp(fullfile(PathName,FileName))
                disp('File already exists, skip')
            end
        end
    end
    disp('Done')
end