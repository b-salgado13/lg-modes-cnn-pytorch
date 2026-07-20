classdef GUILaguerreGaussBeams < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure              matlab.ui.Figure
        NetTypeDropDown       matlab.ui.control.DropDown
        NetTypeDropDownLabel  matlab.ui.control.Label
        Label3                matlab.ui.control.Label
        Image                 matlab.ui.control.Image
        Image3                matlab.ui.control.Image
        FibraOptica           matlab.ui.control.Image
        Label2                matlab.ui.control.Label
        Label                 matlab.ui.control.Label
        PredictLGModeButton   matlab.ui.control.Button
        SelectPatternButton   matlab.ui.control.StateButton
        PredictedLGModeLabel  matlab.ui.control.Label
        OriginalPatternLabel  matlab.ui.control.Label
        Image2                matlab.ui.control.Image
        Fondo                 matlab.ui.control.Image
    end

    % Callbacks that handle component events
    methods (Access = private)

        % Code that executes after component creation
        function startupFcn(app)
            global net
            app.Fondo.ImageSource = imread('Fondo-1.jpg');
            app.Fondo.Visible = 'on';
            app.FibraOptica.Visible = 'off';
            app.Image2.Visible = 'off';
            app.Image.Visible = 'off';
            app.Image.Position = [76,515,550,128];
            app.Image3.Visible = 'off';
            app.Label.Text = "";
            TipoRed = app.NetTypeDropDown.Value;
            if strcmp(TipoRed, 'Convolucional Simple')
                net = load('CNN-8-8-8-Nuevo-9860.mat');
                app.Label2.Text = "Dimension required: 128x192x1";
            else
                net = load('Resnet18-50FL-Nuevo-9316.mat');
                app.Label2.Text = "Dimension required: 224x224x3";
            end
            app.Label2.FontWeight = 'bold';
            app.Label2.FontColor = 'w';
            app.Label2.FontSize = 15;
            app.Label3.Text = "";
            app.PredictLGModeButton.Visible = "off";
            app.PredictedLGModeLabel.FontColor = 'w';
            app.OriginalPatternLabel.FontColor = 'w';
            app.NetTypeDropDownLabel.FontColor = 'w';

        end

        % Callback function
        function Image2Clicked(app, event)

        end

        % Value changed function: SelectPatternButton
        function SelectPatternButtonValueChanged(app, event)
            global OriginalPattern1 u im
            app.SelectPatternButton.BackgroundColor = 'g';
            pause(0.1)
            app.SelectPatternButton.BackgroundColor = 'c';
            startupFcn(app)
            [filename, pathname] = uigetfile({'*.jpg'},'File Selector');
            fullpathname = strcat(pathname,filename);
            OriginalPattern1 = imread(fullpathname);
            tama = size(OriginalPattern1);
            TipoRed = app.NetTypeDropDown.Value;
            if strcmp(TipoRed, 'Convolucional Simple')
                dimension = [128 192];
            else
                dimension = [224 224 3];
            end
            subdim = size(dimension);
            if size(tama) == subdim
                if tama == dimension
                    app.Fondo.ImageSource = imread('Fondo-2.jpg');
                    app.Image.Visible = 'on';
                    app.PredictedLGModeLabel.FontColor = 'b';
                    app.OriginalPatternLabel.FontColor = 'b';
                    app.NetTypeDropDownLabel.FontColor = 'b';
                    app.Label2.FontColor = 'b';
                    app.SelectPatternButton.Enable = "off";
                    app.NetTypeDropDown.Enable = "off";
                    modo = regexp(pathname,'\_','split');
                    type = regexp(char(modo(end-1)),'\/','split');
                    if size(char(type(end)),2) == 2
                        modo = regexp(char(modo(end)),'\/','split');
                        modo = char(modo(1));
                        p = regexp(modo(end),'\d','match');
                        if size(char(regexp(modo,'-\d','match')),1) == 1
                            m = '-1';
                        else
                            m = '1';
                        end
                        app.Label2.Text = strcat('Free space: m = ', m, ',p = ', p);
                        modoname = strcat('lg_', modo,'.JPG');
                    else
                        modo = regexp(char(modo(end)),'\/','split');
                        modo = char(modo(1));
                        app.Label2.Text = strcat('Optic Fiber: m = ', modo, ',p = 0');
                        modoname = strcat('im_', modo,'.JPG');
                    end
                    OriginalPattern0 = imread(strcat(pathname, modoname));
                    if strcmp(TipoRed, 'Convolucional Simple')
                        OrigPatt0 = cat(3, OriginalPattern0, OriginalPattern0, OriginalPattern0);
                        OrigPatt1 = cat(3, OriginalPattern1, OriginalPattern1, OriginalPattern1);
                    else 
                        OrigPatt0 = OriginalPattern0;
                        OrigPatt1 = OriginalPattern1;
                    end
                    app.Image.ImageSource = OrigPatt0;
                    pause(5)
                    app.Image3.ImageSource = OrigPatt0;
                    app.Image3.Position = [40 515 192 128];
                    app.Image3.Visible = 'on';
                    app.FibraOptica = uiimage(app.UIFigure);
                    app.FibraOptica.Position = [254,515,188,60];
                    app.FibraOptica.ScaleMethod = 'fit';
                    app.FibraOptica.ImageSource = 'fibra optica.png';
                    app.FibraOptica.Visible = 'on';
                    h = 250;
                    t = 0.01;
                    for i = 1:h
                        de = 3*i/220;
                        app.Image.Position = [40+i 515 192/log2(2+ de) 128/log2(2+ de)];
                        pause(t)
                    end
                    app.Image.ImageSource = OrigPatt1;
                    norm = log2(2 + 3*h/220);
                    for i = 1:h
                        de = 3*i/220;
                        app.Image.Position = [45+h+i/1.5 515 192*log2(2+de)/norm 128*log2(2+de)/norm];
                        pause(t)
                    end
                    app.PredictLGModeButton.Visible = "on";
                    app.PredictLGModeButton.Enable = "on";
                else
                    app.Label2.FontColor = 'r';
                    app.Label2.FontWeight = 'bold';
                    app.Label2.FontSize = 17;
                end
            else
                app.Label2.FontColor = 'r';
                app.Label2.FontWeight = 'bold';
                app.Label2.FontSize = 17;
            end
            app.SelectPatternButton.Enable = "on";
            app.NetTypeDropDown.Enable = "on";
            app.SelectPatternButton.BackgroundColor = [0.96,0.96,0.96];
        end

        % Button pushed function: PredictLGModeButton
        function PredictLGModeButtonPushed(app, event)
            global OriginalPattern1 net
            app.PredictLGModeButton.BackgroundColor = 'g';
            pause(0.1)
            app.PredictLGModeButton.BackgroundColor = 'c';
            LGMode = string(classify(net.net,OriginalPattern1));
            TipoRed = app.NetTypeDropDown.Value;
            if strcmp(TipoRed, 'Convolucional Simple')
                layer = "softmax";
            else 
                layer = "prob";
            end
            Probabilidad = activations(net.net,OriginalPattern1,layer,OutputAs="rows");
            prob = fix(max(Probabilidad)*100*10^2)/10^2;
            modo = regexp(LGMode,'\_','split');
            if char(modo(1)) == "LG"
                p = regexp(modo(end),'\d','match');
                if size(char(regexp(modo(end),'-\d','match')),1) == 1
                    texto = strcat(['Free space:' ...
                        ' m = -1, p = '],p(end));
                else
                    texto = strcat(['Free space: m = 1, p = '],p(end));
                end
                ImageDir = strcat("LG_",modo(end),".png");
            else
                texto = strcat(['Optic Fiber: m = '],modo(end), ',p = 0');
                ImageDir = strcat("im_",modo(end),".JPG");
            end
            app.Label.Text = texto;
            app.Label3.Text = strcat('Probabilidad = ', string(prob), '%');
            PredictedLGMMode = imread(ImageDir);
            app.Image2.Visible = 'on';
            app.Image2.ImageSource = PredictedLGMMode;
            app.PredictLGModeButton.Enable = "off";
        end

        % Value changed function: NetTypeDropDown
        function NetTypeDropDownValueChanged(app, event)
            startupFcn(app);
        end
    end
    % Component initialization
    methods (Access = private)


    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = GUILaguerreGaussBeams

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            % Execute the startup function
            runStartupFcn(app, @startupFcn)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end
