function str = makefilename(varargin)

str = [];

for i = 1:nargin
   var = varargin{i};
   if isfloat(var)
       var = num2str(var);
       var =  strrep(var, '.', '-');
   end
   str = [str, var];
end

str = [str, '.mat'];
   
   
