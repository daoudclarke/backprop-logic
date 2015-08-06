require 'torch'
require 'nn'
local Term, parent = torch.class('Term', 'nn.Module')

function Term:__init(num_entities)
   parent.__init(self)
   
   self.size = num_entities
  
   self.weight = torch.Tensor(self.size, 1)
   self.gradWeight = torch.Tensor(self.size, 1)
   
   self.output:resize(self.size) 

   self:reset()
end
 
function Term:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:nElement())
   end
   self.weight:uniform(-stdv,stdv)
end

function Term:updateOutput(input)
   self.output = self.weight   
   -- print("Output:", self.output)
   return self.output
end

-- function Term:updateGradInput(input, gradOutput)
--    if not self.gradInput then
--       return
--    end
   
--    self._gradOutput = self._gradOutput or input.new()
--    self._gradInput = self._gradInput or input.new()

--    self.gradInput:resizeAs(input):zero()
--    if self.weight:nElement() == gradOutput:nElement() then
--       self.gradInput:addcmul(1, self.weight, gradOutput)
--    else
--       local batchSize = input:size(1)
--       self._gradOutput:view(gradOutput, batchSize, -1)
--       self._gradInput:view(self.gradInput, batchSize, -1)
--       self._weight:view(self.weight, 1, -1)
--       self._expand:expandAs(self._weight, self._gradOutput)
      
--       if torch.type(input) == 'torch.CudaTensor' then
--          self._repeat:resizeAs(self._expand):copy(self._expand)
--          self._gradInput:addcmul(1, self._repeat, self._gradOutput)
--       else
--          self._gradInput:addcmul(1, self._expand, self._gradOutput)
--       end
--    end
   
--    return self.gradInput
-- end

function Term:accGradParameters(input, gradOutput, scale)
   -- print("Grad output:", gradOutput)
   scale = scale or 1
   self.gradWeight:add(-1*scale, gradOutput)

   
   -- self._input = self._input or input.new()
   -- self._gradWeight = self._gradWeight or input.new()
   -- self._sum = self._sum or input.new()
   
   -- if self.weight:nElement() == gradOutput:nElement() then
   --    self.gradWeight:addcmul(scale, input, gradOutput)
   -- else
   --    local batchSize = input:size(1)
   --    self._input:view(input, batchSize, -1)
   --    self._gradOutput:view(gradOutput, batchSize, -1)
   --    self._gradWeight:view(self.gradWeight, 1, -1)
      
   --    self._repeat:cmul(self._input, self._gradOutput)
   --    self._sum:sum(self._repeat, 1)
   --    self._gradWeight:add(scale, self._sum)
   -- end
end

function Term:type(type)
   if type then
      self._input = nil
      self._output = nil
      self._weight = nil
      self._gradWeight = nil
      self._expand = nil
      self._repeat = nil
      self._sum = nil
   end
   return parent.type(self, type)
end


function Term.__tostring__(self)
   return tostring(self.weight)
end
