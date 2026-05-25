% PlotAverageSpectrumFromSVD_Batch
% Batch-processes every *.svd under data/ (and subfolders) — saves one
% NPZ + PNG per file in Training_data/.
%
% New naming convention (parsed automatically):
%   Folder : r<N>_<x_mm>_<y_mm>_<z_mm>_w<weight_g>
%   File   : <MM>_<DD>_r<N>_<axis>_<burial_pct>_<face>_<run>.svd
%              axis  = x (horizontal scan) | z (vertical into sand)
%              face  = which edge of the rock faces the scanner
%              b/s   = buried / surface
%
% NPZ stores spectrum + all parsed metadata fields.
% PNG title shows rock dimensions, weight, and measurement setup.
%
% Usage:
%   PlotAverageSpectrumFromSVD_Batch
%   PlotAverageSpectrumFromSVD_Batch(dataDir, frame)

function PlotAverageSpectrumFromSVD_Batch(dataDir, frame)
    here = fileparts(mfilename('fullpath'));
    if isempty(here), here = pwd; end
    addpath(here);

    if nargin < 1 || isempty(dataDir)
        dataDir = fullfile(here, 'RAW_data');
    end
    if nargin < 2 || isempty(frame)
        frame = 0;
    end

    if ~exist(dataDir, 'dir')
        error('Data folder not found: %s', dataDir);
    end

    % Recurse into subfolders
    listing = dir(fullfile(dataDir, '**', '*.svd'));
    if isempty(listing)
        warning('PlotAverageSpectrumFromSVD_Batch:noSvd', 'No .svd files under %s', dataDir);
        return;
    end

    fprintf('PlotAverageSpectrumFromSVD_Batch: %d file(s) found\n', numel(listing));

    for k = 1:numel(listing)
        fpath = fullfile(listing(k).folder, listing(k).name);
        fprintf('--- [%d/%d] %s\n', k, numel(listing), fpath);
        try
            ProcessOneSvdFile(fpath, frame, here);
        catch ME
            warning('PlotAverageSpectrumFromSVD_Batch:fileFailed', ...
                'Failed for %s:\n%s', fpath, getReport(ME, 'basic'));
        end
    end
    fprintf('Batch finished.\n');
end

% =========================================================================
function ProcessOneSvdFile(filename, frame, outputDir)
    domainname  = 'FFT';
    channelname = 'Vib & Ref1';
    signalname  = 'H1 Velocity / Force';
    displayname = 'Mag. & Phase';

    % --- Parse metadata from folder + file names ---
    [fdir, fname, ~] = fileparts(filename);
    folderName = basename(fdir);
    folderMeta = ParseFolderName(folderName);
    fileMeta   = ParseFileName(fname);

    [x, avgMag, avgPhase, usd] = LocalGetAverageSpectrumFromSVD(...
        filename, domainname, channelname, signalname, displayname, frame, true);

    if isempty(x) || isempty(avgMag)
        error('No averaged data found for %s', filename);
    end

    freqMax  = 400;
    freqMask = x <= freqMax;
    x        = x(freqMask);
    avgMag   = avgMag(freqMask);

    % --- Peak frequency ---
    [peakMag, idxPeak] = max(avgMag);
    freqPeak  = x(idxPeak);
    threshold = peakMag / sqrt(2);

    % --- Half-power bandwidth ---
    hpOk = false;
    fnHp = freqPeak; zeta = NaN; f1 = NaN; f2 = NaN;
    leftMag = avgMag(1:idxPeak);
    iBelow  = find(leftMag < threshold, 1, 'last');
    if ~isempty(iBelow) && iBelow < idxPeak
        iA = iBelow; iB = iBelow + 1;
        f1tmp = x(iA) + (threshold - leftMag(iA)) / (leftMag(iB) - leftMag(iA)) * (x(iB) - x(iA));
        rightMag = avgMag(idxPeak:end);
        jBelow   = find(rightMag < threshold, 1, 'first');
        if ~isempty(jBelow) && jBelow > 1
            absJ = idxPeak + jBelow - 1;
            iC = absJ - 1; iD = absJ;
            f2tmp = x(iC) + (threshold - avgMag(iC)) / (avgMag(iD) - avgMag(iC)) * (x(iD) - x(iC));
            if f2tmp > f1tmp
                f1 = f1tmp; f2 = f2tmp;
                fnHp = (f1 + f2) / 2;
                zeta = (f2 - f1) / (2 * fnHp);
                hpOk = true;
            end
        end
    end
    if ~hpOk
        warning('PlotAverageSpectrumFromSVD_Batch:halfPowerFailed', ...
            'No -3 dB crossings for %s; using peak freq.', filename);
    end

    fprintf('  Peak: %.2f Hz  |  HP fn: %.2f Hz  zeta=%.4f  f1=%.2f  f2=%.2f\n', ...
        freqPeak, fnHp, zeta, f1, f2);

    % --- Averaged phase at resonance (from averaged H1 spectrum) ---
    fnAvg = (freqPeak + fnHp) / 2;   % same composite fn used as feature
    phaseAtFnAvg_deg = NaN;
    if ~isempty(avgPhase) && numel(avgPhase) == numel(x)
        [~, iFn] = min(abs(x - fnAvg));
        phaseAtFnAvg_deg = avgPhase(iFn);
    end

    % --- Per-point phase features ---
    [phaseSpread_deg, phaseSlope_deg_per_mm, phaseFlipFlag, ...
     allPhases_deg, allYmm_ph, nPhasePts] = ...
        LocalGetPhaseFeatures(filename, frame, fnAvg);
    fprintf('  Phase at fn: %.1f deg (avg) | spread=%.1f deg | slope=%.3f deg/mm | flip=%d  (n=%d pts)\n', ...
        phaseAtFnAvg_deg, phaseSpread_deg, phaseSlope_deg_per_mm, phaseFlipFlag, nPhasePts);

    % --- Spatial gradient (linear slope across all scan points) ---
    [spatialRatio, spatialSlope, spatialR2, ...
     topMag, bottomMag, yTopMm, yBottomMm, ...
     allMags, allYmm, nScanPts] = LocalGetSpatialGradient(filename, frame, freqPeak);
    fprintf('  Spatial: top=%.4g  bot=%.4g  ratio=%.3f  slope=%.4g/mm  R2=%.2f  (n=%d pts)\n', ...
        topMag, bottomMag, spatialRatio, spatialSlope, spatialR2, nScanPts);

    % --- Output paths ---
    trainingDir = fullfile(outputDir, '..', 'NPZ_data');
    if ~exist(trainingDir, 'dir'), mkdir(trainingDir); end

    safeStem = regexprep(fname, '[^a-zA-Z0-9_\-]', '_');
    freqTag  = regexprep(sprintf('%.2f', fnHp), '\.', 'p');
    baseName = sprintf('%s_fn%sHz_frame_%d', safeStem, freqTag, frame);

    % --- Save NPZ ---
    npzPath = fullfile(trainingDir, [baseName '.npz']);
    try
        LocalSaveSpectrumNpz(x, avgMag, npzPath, fnHp, f1, f2, zeta, freqPeak, hpOk, ...
            folderMeta, fileMeta, spatialRatio, spatialSlope, spatialR2, ...
            topMag, bottomMag, yTopMm, yBottomMm, allMags, allYmm, nScanPts, ...
            phaseAtFnAvg_deg, phaseSpread_deg, phaseSlope_deg_per_mm, phaseFlipFlag, ...
            allPhases_deg, allYmm_ph, nPhasePts);
    catch ME_npz
        warning('PlotAverageSpectrumFromSVD_Batch:npzFailed', ...
            'NPZ export failed for %s:\n%s', filename, getReport(ME_npz, 'basic'));
    end

    % --- Save PNG ---
    pngPath = fullfile(trainingDir, [baseName '.png']);
    try
        LocalSaveSpectrumPng(x, avgMag, pngPath, signalname, usd, ...
            peakMag, freqPeak, threshold, fnHp, f1, f2, zeta, hpOk, ...
            folderMeta, fileMeta);
    catch ME_png
        warning('PlotAverageSpectrumFromSVD_Batch:pngFailed', ...
            'PNG export failed for %s:\n%s', filename, getReport(ME_png, 'basic'));
    end
end

% =========================================================================
% Parse folder name: r<N>_<x>_<y>_<z>_w<W>
function m = ParseFolderName(folderName)
    m = struct('rock_num', NaN, 'size_x_mm', NaN, 'size_y_mm', NaN, ...
               'size_z_mm', NaN, 'weight_g', NaN);
    tok = regexp(folderName, ...
        '^r(\d+)_([\d.]+)_([\d.]+)_([\d.]+)_w(\d+)$', 'tokens', 'once');
    if ~isempty(tok)
        m.rock_num  = str2double(tok{1});
        m.size_x_mm = str2double(tok{2});
        m.size_y_mm = str2double(tok{3});
        m.size_z_mm = str2double(tok{4});
        m.weight_g  = str2double(tok{5});
    else
        % Fallback: extract rock number only
        tok2 = regexp(folderName, '^r(\d+)', 'tokens', 'once');
        if ~isempty(tok2)
            m.rock_num = str2double(tok2{1});
        end
        warning('PlotAverageSpectrumFromSVD_Batch:folderParse', ...
            'Could not fully parse folder name: %s', folderName);
    end
end

% =========================================================================
% Parse file name: <MM>_<DD>_r<N>_<axis>_<burial_pct>_<face>_<run>
function m = ParseFileName(fname)
    m = struct('month', NaN, 'day', NaN, 'rock_num', NaN, ...
               'axis', '', 'burial_pct', NaN, 'face', '', 'run_num', NaN);
    tok = regexp(fname, ...
        '^(\d{2})_(\d{2})_r(\d+)_([xzXZ])_(\d+)_([a-zA-Z]+)_(\d+)$', ...
        'tokens', 'once');
    if ~isempty(tok)
        m.month      = str2double(tok{1});
        m.day        = str2double(tok{2});
        m.rock_num   = str2double(tok{3});
        m.axis       = lower(tok{4});
        m.burial_pct = str2double(tok{5});
        m.face       = lower(tok{6});
        m.run_num    = str2double(tok{7});
    else
        warning('PlotAverageSpectrumFromSVD_Batch:fileParse', ...
            'Could not parse file name: %s', fname);
    end
end

% =========================================================================
function LocalSaveSpectrumPng(x, avgMag, pngPath, signalname, usd, ...
        peakMag, freqPeak, threshold, fnHp, f1, f2, zeta, hpOk, ...
        folderMeta, fileMeta)

    freqMax = max(x);
    yMax    = peakMag * 1.15;
    xMin    = min(x);

    fig = figure('Color', 'w', 'Visible', 'off', 'Position', [100 100 1600 700]);

    plot(x, avgMag, 'b-', 'LineWidth', 1.0);
    hold on;
    plot([xMin, freqMax], [threshold, threshold], '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.0);
    plot([freqPeak, freqPeak], [0, yMax], 'r--', 'LineWidth', 1.4);

    if hpOk
        plot([f1, f1], [0, yMax], '--', 'Color', [1 0.45 0], 'LineWidth', 1.2);
        plot([f2, f2], [0, yMax], '--', 'Color', [0.85 0.35 0], 'LineWidth', 1.2);
        plot([fnHp, fnHp], [0, yMax], 'g--', 'LineWidth', 1.35);
        infoStr = sprintf(['Peak: %.1f Hz, %.4g %s (red) | f1,f2: %.1f, %.1f Hz (orange, -3 dB) | ', ...
            'HP f_n: %.1f Hz (green) | zeta=%.3f'], freqPeak, peakMag, usd.YUnit, f1, f2, fnHp, zeta);
    else
        infoStr = sprintf('Peak: %.1f Hz, %.4g %s (red) — half-power crossings not found', ...
            freqPeak, peakMag, usd.YUnit);
    end
    text(xMin + (freqMax - xMin)*0.01, yMax*0.98, infoStr, ...
        'FontSize', 8, 'Color', 'k', 'VerticalAlignment', 'top', 'Interpreter', 'none');

    grid on;
    xlabel(['Frequency (' usd.XUnit ')']);
    ylabel(['Magnitude (' usd.YUnit ')']);

    % Build title from metadata
    titleStr = BuildTitleStr(signalname, folderMeta, fileMeta);
    title(titleStr, 'Interpreter', 'none');

    xlim([xMin, freqMax]);
    ylim([0, yMax]);
    pbaspect([2 1 1]);

    exportgraphics(fig, pngPath, 'Resolution', 300);
    fprintf('  Saved PNG: %s\n', pngPath);
    close(fig);
end

% =========================================================================
function s = BuildTitleStr(signalname, fm, pm)
    % Rock info from folder
    if ~isnan(fm.rock_num)
        rockStr = sprintf('Rock r%d', fm.rock_num);
    else
        rockStr = 'Rock ?';
    end
    if ~isnan(fm.size_x_mm)
        sizeStr = sprintf('%.4g x %.4g x %.4g mm, %.4g g', ...
            fm.size_x_mm, fm.size_y_mm, fm.size_z_mm, fm.weight_g);
    else
        sizeStr = '';
    end

    % Measurement info from file
    if ~isempty(pm.axis)
        axisStr = sprintf('%s-axis', upper(pm.axis));
    else
        axisStr = '';
    end
    if ~isnan(pm.burial_pct)
        burialStr = sprintf('%d%% burial', pm.burial_pct);
    else
        burialStr = '';
    end
    if ~isempty(pm.face)
        faceStr = sprintf('face: %s', pm.face);
    else
        faceStr = '';
    end
    if ~isnan(pm.month)
        dateStr = sprintf('%02d/%02d', pm.month, pm.day);
    else
        dateStr = '';
    end

    parts = {signalname, rockStr, sizeStr, axisStr, burialStr, faceStr, dateStr};
    parts = parts(~cellfun(@isempty, parts));
    s = strjoin(parts, '  |  ');
end

% =========================================================================
function LocalSaveSpectrumNpz(freqHz, magnitude, npzPath, fnHp, f1, f2, zeta, ...
        freqPeak, hpOk, folderMeta, fileMeta, ...
        spatialRatio, spatialSlope, spatialR2, ...
        topMag, bottomMag, yTopMm, yBottomMm, allMags, allYmm, nScanPts, ...
        phaseAtFnAvg_deg, phaseSpread_deg, phaseSlope_deg_per_mm, phaseFlipFlag, ...
        allPhases_deg, allYmm_ph, nPhasePts)
    if nargin < 9,  hpOk                  = false;  end
    if nargin < 10, folderMeta            = struct(); end
    if nargin < 11, fileMeta              = struct(); end
    if nargin < 12, spatialRatio          = NaN;    end
    if nargin < 13, spatialSlope          = NaN;    end
    if nargin < 14, spatialR2             = NaN;    end
    if nargin < 15, topMag                = NaN;    end
    if nargin < 16, bottomMag             = NaN;    end
    if nargin < 17, yTopMm                = NaN;    end
    if nargin < 18, yBottomMm             = NaN;    end
    if nargin < 19, allMags               = [];     end
    if nargin < 20, allYmm                = [];     end
    if nargin < 21, nScanPts              = NaN;    end
    if nargin < 22, phaseAtFnAvg_deg      = NaN;    end
    if nargin < 23, phaseSpread_deg       = NaN;    end
    if nargin < 24, phaseSlope_deg_per_mm = NaN;    end
    if nargin < 25, phaseFlipFlag         = NaN;    end
    if nargin < 26, allPhases_deg         = [];     end
    if nargin < 27, allYmm_ph             = [];     end
    if nargin < 28, nPhasePts             = NaN;    end

    tmpCsv  = [tempname, '.csv'];
    tmpJson = [tempname, '.json'];
    writematrix([freqHz(:), magnitude(:)], tmpCsv, 'Delimiter', ',');

    % Build metadata JSON
    meta.fn_halfpower_Hz  = fnHp;
    meta.f1_Hz            = LocalNanSafe(f1);
    meta.f2_Hz            = LocalNanSafe(f2);
    meta.damping_ratio    = LocalNanSafe(zeta);
    meta.fn_peak_Hz       = freqPeak;
    meta.halfpower_valid  = double(hpOk);
    % Folder metadata
    meta.rock_num         = LocalNanSafe(folderMeta, 'rock_num');
    meta.size_x_mm        = LocalNanSafe(folderMeta, 'size_x_mm');
    meta.size_y_mm        = LocalNanSafe(folderMeta, 'size_y_mm');
    meta.size_z_mm        = LocalNanSafe(folderMeta, 'size_z_mm');
    meta.weight_g         = LocalNanSafe(folderMeta, 'weight_g');
    % File metadata
    meta.month            = LocalNanSafe(fileMeta, 'month');
    meta.day              = LocalNanSafe(fileMeta, 'day');
    meta.axis             = LocalStrSafe(fileMeta, 'axis');
    meta.burial_pct       = LocalNanSafe(fileMeta, 'burial_pct');
    meta.face             = LocalStrSafe(fileMeta, 'face');
    meta.run_num          = LocalNanSafe(fileMeta, 'run_num');
    % Spatial gradient — scalars
    meta.mag_spatial_ratio = LocalNanSafe(spatialRatio);
    meta.spatial_slope     = LocalNanSafe(spatialSlope);
    meta.spatial_r2        = LocalNanSafe(spatialR2);
    meta.mag_top           = LocalNanSafe(topMag);
    meta.mag_bottom        = LocalNanSafe(bottomMag);
    meta.y_top_mm          = LocalNanSafe(yTopMm);
    meta.y_bottom_mm       = LocalNanSafe(yBottomMm);
    meta.n_scan_points     = LocalNanSafe(nScanPts);
    % Per-point magnitude arrays (variable length 3-5) — stored as lists in JSON
    if isempty(allMags), meta.point_mags_sorted = 'null';
    else, meta.point_mags_sorted = allMags(:)'; end
    if isempty(allYmm),  meta.point_ymm_sorted  = 'null';
    else, meta.point_ymm_sorted  = allYmm(:)';  end
    % Phase features
    meta.phase_at_fn_avg_deg      = LocalNanSafe(phaseAtFnAvg_deg);
    meta.phase_spread_deg         = LocalNanSafe(phaseSpread_deg);
    meta.phase_slope_deg_per_mm   = LocalNanSafe(phaseSlope_deg_per_mm);
    meta.phase_flip_flag          = LocalNanSafe(phaseFlipFlag);
    meta.n_phase_points           = LocalNanSafe(nPhasePts);
    % Per-point phase arrays
    if isempty(allPhases_deg), meta.point_phases_deg = 'null';
    else, meta.point_phases_deg = allPhases_deg(:)'; end
    if isempty(allYmm_ph),     meta.point_ymm_ph     = 'null';
    else, meta.point_ymm_ph    = allYmm_ph(:)';      end

    jsonStr = jsonencode(meta);
    fid = fopen(tmpJson, 'w');
    if fid < 0, error('Could not write temp JSON.'); end
    fprintf(fid, '%s', jsonStr);
    fclose(fid);

    pyScript = fullfile(tempdir, 'save_spectrum_npz_v2.py');
    fid = fopen(pyScript, 'w');
    if fid < 0, error('Could not write temporary Python script.'); end
    fprintf(fid, 'import numpy as np, sys, json\n');
    fprintf(fid, 'd = np.loadtxt(sys.argv[1], delimiter='','', ndmin=2)\n');
    fprintf(fid, 'meta = json.loads(open(sys.argv[3]).read())\n');
    fprintf(fid, 'arrays = {}\n');
    fprintf(fid, 'for k, v in meta.items():\n');
    fprintf(fid, '    if v is None or v == ''null'':\n');
    fprintf(fid, '        arrays[k] = np.array(float(''nan''))\n');
    fprintf(fid, '    elif isinstance(v, list):\n');
    fprintf(fid, '        arrays[k] = np.array(v, dtype=float)\n');
    fprintf(fid, '    else:\n');
    fprintf(fid, '        arrays[k] = np.array(v)\n');
    fprintf(fid, 'np.savez(sys.argv[2], frequency_Hz=d[:,0], magnitude=d[:,1], **arrays)\n');
    fclose(fid);

    cmd = sprintf('python "%s" "%s" "%s" "%s"', pyScript, tmpCsv, npzPath, tmpJson);
    [status, msg] = system(cmd);
    if exist(tmpCsv,  'file'), delete(tmpCsv);  end
    if exist(tmpJson, 'file'), delete(tmpJson); end
    if status ~= 0
        warning('PlotAverageSpectrumFromSVD_Batch:npzFailed', 'NPZ export failed.\n%s', msg);
    else
        fprintf('  Saved NPZ: %s\n', npzPath);
    end
end

% =========================================================================
function v = LocalNanSafe(structOrVal, field)
    if nargin == 1
        v = structOrVal;
        if isnan(v), v = 'null'; end
        return;
    end
    if isfield(structOrVal, field)
        v = structOrVal.(field);
        if isnumeric(v) && isnan(v), v = 'null'; end
    else
        v = 'null';
    end
end

function s = LocalStrSafe(st, field)
    if isfield(st, field) && ~isempty(st.(field))
        s = st.(field);
    else
        s = '';
    end
end

function name = basename(p)
    parts = strsplit(p, {'/','\'});
    parts = parts(~cellfun(@isempty, parts));
    name = parts{end};
end

% =========================================================================
% LocalGetPhaseFeatures
%   Opens the SVD and reads per-point phase at the identified resonance fn.
%   Phase is in degrees (Polytec PSV output).
%
%   Outputs
%   -------
%   phaseSpread_deg        std of per-point phase at fn  [deg]
%   phaseSlope_deg_per_mm  linear fit: phase vs Y        [deg/mm]
%   phaseFlipFlag          1 if any adjacent pair differs by >90 deg
%   allPhases_deg          per-point phase at fn, sorted bottom->top
%   allYmm                 corresponding Y coordinates   [mm]
%   nPoints                number of valid scan points
%
function [phaseSpread_deg, phaseSlope_deg_per_mm, phaseFlipFlag, ...
          allPhases_deg, allYmm, nPoints] = ...
        LocalGetPhaseFeatures(filename, frame, fnHz)

    phaseSpread_deg       = NaN;
    phaseSlope_deg_per_mm = NaN;
    phaseFlipFlag         = NaN;
    allPhases_deg         = [];
    allYmm                = [];
    nPoints               = 0;

    domainname  = 'FFT';
    channelname = 'Vib & Ref1';
    signalname  = 'H1 Velocity / Force';
    displayname = 'Mag. & Phase';

    file = [];
    try
        file = actxserver('PolyFile.PolyFile');
        file.Open(filename);

        pd  = file.GetPointDomains();
        dom = pd.Item(domainname);
        ch  = dom.Channels.Item(channelname);
        sig = ch.Signals.Item(signalname);
        disp_obj = sig.Displays.Item(displayname);

        sdesc = sig.Description;
        xax   = sdesc.XAxis;
        nFreq = xax.MaxCount;
        freqVec = linspace(xax.Min, xax.Max, nFreq);

        dps     = dom.datapoints;
        nPoints = dps.count;

        if nPoints < 2
            file.Close(); delete(file); return;
        end

        yCoords    = nan(1, nPoints);
        phaseAtFn  = nan(1, nPoints);

        % Window: ±2% around fn
        winHalf = max(1, round(nFreq * 0.02));
        [~, iFn] = min(abs(freqVec - fnHz));
        iLo = max(1, iFn - winHalf);
        iHi = min(nFreq, iFn + winHalf);

        for i = 1:nPoints
            dp = dps.Item(i);
            mp = dp.MeasPoint;

            ay = mp.ScanQuantitiesY;
            d  = mp.ScanDistances;
            yCoords(i) = d * tand(ay) * 1e3;

            raw = double(dp.GetData(disp_obj, frame));
            if isempty(raw), continue; end

            % Mag. & Phase interleaved: odd = magnitude, even = phase [deg]
            if numel(raw) >= 2 * nFreq
                phVec = raw(2:2:2*nFreq);
            else
                continue   % no phase data for this point
            end

            % Phase at the frequency of max magnitude within window
            % (avoids picking phase where magnitude is noise-floor)
            magVec = raw(1:2:2*nFreq);
            [~, iPk] = max(magVec(iLo:iHi));
            iPk = iLo + iPk - 1;
            phaseAtFn(i) = phVec(iPk);
        end

        file.Close(); delete(file);

        % Sort by Y (ascending = bottom to top)
        [sortedY, sortIdx] = sort(yCoords, 'ascend');
        sortedPh = phaseAtFn(sortIdx);

        valid = ~isnan(sortedPh) & ~isnan(sortedY);
        if sum(valid) < 2, return; end

        allYmm        = sortedY(valid);
        allPhases_deg = sortedPh(valid);
        nPoints       = numel(allYmm);

        % Phase spread
        phaseSpread_deg = std(allPhases_deg);

        % Phase slope vs height (deg/mm)
        if nPoints >= 2
            p = polyfit(allYmm, allPhases_deg, 1);
            phaseSlope_deg_per_mm = p(1);
        end

        % Phase flip: check if any adjacent pair (sorted by height)
        % differs by more than 90 degrees (mod 360 wrapping)
        diffs = diff(allPhases_deg);
        % Wrap to [-180, 180]
        diffs = mod(diffs + 180, 360) - 180;
        phaseFlipFlag = double(any(abs(diffs) > 90));

    catch ME
        warning('PlotAverageSpectrumFromSVD_Batch:phaseFeatures', ...
            'LocalGetPhaseFeatures failed for %s:\n%s', filename, ME.message);
        try
            if ~isempty(file) && isvalid(file)
                file.Close(); delete(file);
            end
        catch, end
    end
end

% =========================================================================
function [x, avgMag, avgPhase, usd] = LocalGetAverageSpectrumFromSVD( ...
        filename, domainname, channelname, signalname, displayname, frame, usePhase)
    if nargin < 1 || isempty(filename),    filename    = 'g:/My Drive/Rock_Vib/02_25_bigRock_50_b.svd'; end
    if nargin < 2 || isempty(domainname),  domainname  = 'FFT';               end
    if nargin < 3 || isempty(channelname), channelname = 'Vib & Ref1';        end
    if nargin < 4 || isempty(signalname),  signalname  = 'H1 Velocity / Force'; end
    if nargin < 5 || isempty(displayname), displayname = 'Mag. & Phase';      end
    if nargin < 6 || isempty(frame),       frame       = 0;                   end

    allPoints = 0;
    [x, y, usd] = GetPointData(filename, domainname, channelname, signalname, displayname, allPoints, frame);

    if nargin < 7 || isempty(usePhase)
        usePhase = (size(y, 2) >= 2 && mod(size(y, 2), 2) == 0);
    end

    if isempty(y) || numel(y) == 0
        x = []; avgMag = []; avgPhase = []; return;
    end

    validRows = any(abs(y) > 0, 2);
    if ~any(validRows), validRows = true(size(y, 1), 1); end
    yValid = y(validRows, :);

    hasAltPairs = (size(yValid, 2) >= 2) && (mod(size(yValid, 2), 2) == 0);
    if hasAltPairs
        magRows = yValid(:, 1:2:end);
        if usePhase
            phaseRows = yValid(:, 2:2:end);
            avgMag    = mean(magRows,   1, 'omitnan');
            avgPhase  = mean(phaseRows, 1, 'omitnan');
        else
            avgMag   = mean(magRows, 1, 'omitnan');
            avgPhase = [];
        end
        x = x(1:numel(avgMag));
    else
        avgMag   = mean(yValid, 1, 'omitnan');
        avgPhase = [];
        if usePhase
            warning('No paired phase data. Returning magnitude only.');
        end
    end

    x        = x(:)';
    avgMag   = avgMag(:)';
    if ~isempty(avgPhase), avgPhase = avgPhase(:)'; end
end

% =========================================================================
% LocalGetSpatialGradient
%   Opens the SVD file, retrieves per-point FRF peak magnitude and Y
%   position for every measurement point on the rock face.
%
%   Points are sorted by Y (ascending = bottom to top):
%     bottom = lowest Y point   (closest to / in sand)
%     top    = top 2 Y points   (furthest from sand)
%     middle = everything in between (0, 1, or 2 points)
%
%   Outputs
%   -------
%   spatialRatio   top2_mean / bottom peak mag       (unit-less ratio)
%   spatialSlope   linear-fit slope of mag vs Y      [(FRF unit)/mm]
%                  fitted across ALL valid points — uses middle points too
%                  Negative = mag decreases toward sand (expected)
%   spatialR2      R² of the linear fit (quality indicator, 0-1)
%   topMag         mean peak mag of top 2 points
%   bottomMag      peak mag of bottom point
%   yTopMm         mean Y coordinate of top 2 points   [mm]
%   yBottomMm      Y coordinate of bottom point         [mm]
%   allMags        peak mags sorted bottom->top (length = nPoints)
%   allYmm         Y coordinates sorted bottom->top     [mm]
%   nPoints        total number of scan points
%
function [spatialRatio, spatialSlope, spatialR2, ...
          topMag, bottomMag, yTopMm, yBottomMm, ...
          allMags, allYmm, nPoints] = ...
        LocalGetSpatialGradient(filename, frame, peakFreq)

    spatialRatio = NaN; spatialSlope = NaN; spatialR2 = NaN;
    topMag = NaN; bottomMag = NaN; yTopMm = NaN; yBottomMm = NaN;
    allMags = []; allYmm = []; nPoints = 0;

    domainname  = 'FFT';
    channelname = 'Vib & Ref1';
    signalname  = 'H1 Velocity / Force';
    displayname = 'Mag. & Phase';

    file = [];
    try
        file = actxserver('PolyFile.PolyFile');
        file.Open(filename);

        pd  = file.GetPointDomains();
        dom = pd.Item(domainname);
        ch  = dom.Channels.Item(channelname);
        sig = ch.Signals.Item(signalname);
        disp_obj = sig.Displays.Item(displayname);

        % Build frequency axis
        sdesc = sig.Description;
        xax   = sdesc.XAxis;
        nFreq = xax.MaxCount;
        freqVec = linspace(xax.Min, xax.Max, nFreq);

        dps     = dom.datapoints;
        nPoints = dps.count;

        if nPoints < 3
            file.Close(); delete(file); return;
        end

        yCoords  = nan(1, nPoints);
        peakMags = nan(1, nPoints);

        % Window around the peak frequency: +/- 2% of full range
        winHalf = max(1, round(nFreq * 0.02));

        for i = 1:nPoints
            dp = dps.Item(i);
            mp = dp.MeasPoint;

            % Y coordinate (mm)
            ay = mp.ScanQuantitiesY;  % angle in degrees
            d  = mp.ScanDistances;    % stand-off distance (mm)
            yCoords(i) = d * tand(ay) * 1e3;

            % Magnitude spectrum for this point
            raw = double(dp.GetData(disp_obj, frame));
            if isempty(raw), continue; end

            if numel(raw) >= 2 * nFreq
                magVec = raw(1:2:2*nFreq);
            elseif numel(raw) >= nFreq
                magVec = raw(1:nFreq);
            else
                magVec = raw;
            end

            [~, iPeak] = min(abs(freqVec(1:numel(magVec)) - peakFreq));
            iLo = max(1, iPeak - winHalf);
            iHi = min(numel(magVec), iPeak + winHalf);
            peakMags(i) = max(magVec(iLo:iHi));
        end

        file.Close(); delete(file);

        % Sort ascending by Y: index 1 = bottom, end = top
        [sortedY, sortIdx] = sort(yCoords, 'ascend');
        sortedMags = peakMags(sortIdx);

        % Keep only valid (non-NaN) points
        valid = ~isnan(sortedMags) & ~isnan(sortedY);
        if sum(valid) < 2, return; end

        allYmm  = sortedY(valid);
        allMags = sortedMags(valid);
        nValid  = numel(allYmm);

        % ── Top/bottom summary ──────────────────────────────────────────
        bottomMag  = allMags(1);
        yBottomMm  = allYmm(1);
        topMag     = mean(allMags(max(1, nValid-1):end));   % top 2 avg
        yTopMm     = mean(allYmm(max(1, nValid-1):end));

        if bottomMag > 0 && ~isnan(topMag)
            spatialRatio = topMag / bottomMag;
        end

        % ── Linear slope across ALL points: mag = slope * Y + intercept ─
        % Uses polyfit (least squares). slope < 0 means mag decreases
        % toward lower Y (toward sand) — stronger with deeper burial.
        if nValid >= 2
            p   = polyfit(allYmm, allMags, 1);
            spatialSlope = p(1);   % (FRF unit) / mm

            % R²
            yHat = polyval(p, allYmm);
            ssTot = sum((allMags - mean(allMags)).^2);
            ssRes = sum((allMags - yHat).^2);
            if ssTot > 0
                spatialR2 = 1 - ssRes / ssTot;
            else
                spatialR2 = NaN;
            end
        end

    catch ME
        warning('PlotAverageSpectrumFromSVD_Batch:spatialGrad', ...
            'LocalGetSpatialGradient failed for %s:\n%s', filename, ME.message);
        try
            if ~isempty(file) && isvalid(file)
                file.Close(); delete(file);
            end
        catch, end
    end
end
