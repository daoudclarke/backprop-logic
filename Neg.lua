require 'torch'
require 'nn'
local Neg, parent = torch.class('Neg', 'nn.Module')

-- function Neg:__init()
--    parent.__init(self)
  
--    self.weight = torch.Tensor(1)
--    self.gradWeight = torch.Tensor(1)
   
--    self:reset()
-- end

-- function Neg:reset(stdv)
--    self.weight[1] = -1.0
-- end

function Neg:updateOutput(input)
   self.output:resizeAs(input):copy(input);
   self.output:mul(-1.0);
   return self.output
end

function Neg:updateGradInput(input, gradOutput) 
   self.gradInput:resizeAs(input):zero()
   self.gradInput:add(-1.0, gradOutput)
   return self.gradInput
end

-- function Neg:accGradParameters(input, gradOutput, scale) 
--    scale = scale or 1
--    self.gradWeight[1] = self.gradWeight[1] + scale*input:dot(gradOutput);
-- end
