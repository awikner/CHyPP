classdef circularPadLayer < nnet.layer.Layer
    % Example custom PReLU layer.
    properties
        filterSize
    end
    
    properties (Dependent)
        padSize
    end
    
    methods
        function layer = circularPadLayer(filterSize, name)
            
            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = "circularPadLayer";
            
            % Set padding size
            layer.filterSize = filterSize;
        end
        
        function value = get.padSize(layer)
            iseven = double(mod(layer.filterSize,2)==0);
            value = [floor(layer.filterSize/2);iseven];
        end
        
        function Z = predict(layer, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer and outputs the result Z.
            
            Z = padarray(X,layer.padSize(1,:),'circular');
            if layer.padSize(2,1)==1
                Z(end,:) = NaN;
                Z = Z(~isnan(Z));
            end
            if size(layer.padSize,2)>1
                if layer.padSize(2,2)==1
                    Z(:,end) = NaN;
                    Z = Z(~isnan(Z));
                end
            end
        end
        
        function [dLdX] = backward(layer, X, Z, dLdZ, memory)
            % [dLdX, dLdAlpha] = backward(layer, X, Z, dLdZ, memory)
            % backward propagates the derivative of the loss function
            % through the layer.
            %
            % Inputs:
            %         layer    - Layer to backward propagate through 
            %         X        - Input data 
            %         Z        - Output of layer forward function 
            %         dLdZ     - Gradient propagated from the deeper layer 
            %         memory   - Memory value which can be used in backward
            %                    propagation
            % Outputs:
            %         dLdX     - Derivative of the loss with respect to the
            %                    input data
            %         dLdAlpha - Derivative of the loss with respect to the
            %                    learnable parameter Alpha
            
%             dLdX = layer.Alpha .* dLdZ;
%             dLdX(X>0) = dLdZ(X>0);
%             dLdAlpha = min(0,X) .* dLdZ;
%             dLdAlpha = sum(sum(dLdAlpha,1),2);
%             
%             % Sum over all observations in mini-batch.
%             dLdAlpha = sum(dLdAlpha,4);
            dLdX = single(zeros(size(X)));
        end
    end
end