
filename = 'test.txt';

record = [];
fid = fopen(filename,'r');
disp('Reading data...');
while (~feof(fid))
    line = fgetl(fid);
    
    [time,line] = strtok(line);  time = str2num(time);
    [ident,line] = strtok(line); ident = lower(ident);
    data = str2num(line);
    if (isempty(data))
        data = {strtrim(line)};
    end;
    if (strcmp(ident,'annot'))
        time = lasttime;
    end;
    
    if (~isfield(record,ident))
        record.(ident).time = zeros(0,1);
        if (iscell(data))
            record.(ident).data = cell(0,length(data));
        else
            record.(ident).data = zeros(0,length(data));
        end;
    end;
    record.(ident).time = [ record.(ident).time; time ];
    record.(ident).data = [ record.(ident).data; data ];
    
    % Track last time point to assign to future annotations
    lasttime = time;
    
end;
fclose(fid);

disp('Zeroing time vectors...');
fnames = fieldnames(record);
time0 = Inf;
for ind = (1:length(fnames))
    ident = fnames{ind};
    time0 = min([ time0 record.(ident).time(1) ]);
end;
for ind = (1:length(fnames))
    ident = fnames{ind};
    record.(ident).time = record.(ident).time-time0;
end;

% Convert orientation data from unit quarternion to { pitch, yaw, roll } in radians
if (isfield(record,'ori'))
    ori = record.ori;
    record.ori.pitch = asin(max(-1, min(1, 2*(ori.data(:,1) .* ori.data(:,3) - ori.data(:,4) .* ori.data(:,2)))));
    record.ori.yaw = atan2(2*(ori.data(:,1) .* ori.data(:,4) + ori.data(:,2) .* ori.data(:,3)), 1 - 2 * (ori.data(:,3) .* ori.data(:,3) + ori.data(:,4) .* ori.data(:,4)));
    record.ori.roll = atan2(2*(ori.data(:,1) .* ori.data(:,2) + ori.data(:,3) .* ori.data(:,4)), 1 - 2 * (ori.data(:,2) .* ori.data(:,2) + ori.data(:,3) .* ori.data(:,3)));
end;

disp('done.');

% Plot data
if ( 1 )
    figure;
    haxes(1) = subplot(121);
        addpath('C:\Users\brittain\Dropbox\Matlab\library\misc');
        stackplotsc( record.emg.time, record.emg.data );
        title('Electromyography'); xlabel('Time (secs)');
    haxes(2) = subplot(222);
        plot( record.acc.time, record.acc.data ); axis('tight');
        title('Acceleration'); xlabel('Time (secs)');
    haxes(3) = subplot(224);
        plot( record.ori.time, record.ori.pitch*180/pi ); hold('on');
        plot( record.ori.time, record.ori.yaw*180/pi );
        plot( record.ori.time, record.ori.roll*180/pi ); axis('tight');
        legend('Pitch','Yaw','Roll');
        title('Orientation'); xlabel('Time (secs)');
    % Add annotation scheme
    for ind = 1:length(haxes)
        axes(haxes(ind)); hold('on');
        cmap = lines(12);
        for k = (1:length(record.annot.time))
            tmin = record.annot.time(k);
            if ( k == length(record.annot.time) )
                tmax = max(cellfun(@(x) max(record.(x).time),fieldnames(record)));
            else
                tmax = record.annot.time(k+1);
            end;
            [fkey,desc] = strtok(record.annot.data{k},' '); fkey = str2num(fkey(2:end));
            h(k) = fill( [tmin tmin tmax tmax], [ylim fliplr(ylim)] , 'g', 'linestyle', 'none', 'facecolor', cmap(fkey,:), 'facealpha', 0.5 );
            text( tmin, max(ylim), desc, 'color', [1 1 1], 'VerticalAlignment', 'Top', 'HorizontalAlignment', 'Left' );
        end;
        if ( ind == 1 )
            [strLegend,ia,ic] = unique(record.annot.data);
            [~,strLegend] = strtok(strLegend); strLegend = strtrim(strLegend);
            legend(h(ia),strLegend);        % unique returns sorted order
        end;
    end;
end;
