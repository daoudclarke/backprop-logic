require 'torch'
require 'nn'
local Term, parent = torch.class('Term', 'nn.Module')

function Term:__init(num_entities)
   parent.__init(self)
   
   self.size = num_entities
  
   self.weight = torch.Tensor(self.size)
   self.gradWeight = torch.Tensor(self.size)
   
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

-- function Term:updateOutput(input)
--    -- lazy-initialize
--    self._output = self._output or input.new()
--    self._weight = self._weight or input.new()
--    self._expand = self._expand or input.new()
--    self._repeat = self._repeat or input.new()
   
--    self.output:resizeAs(input):copy(input)
--    if input:nElement() == self.weight:nElement() then
--       self._output:view(self.output, -1)
--       self._weight:view(self.weight, -1)
      
--       self._output:cmul(self._weight)
--    else
--       local batchSize = input:size(1)
--       self._output:view(self.output, batchSize, -1)
--       self._weight:view(self.weight, 1, -1)
      
--       self._expand:expandAs(self._weight, self._output)
      
--       if torch.type(input) == 'torch.CudaTensor' then
--          self._repeat:resizeAs(self._expand):copy(self._expand)
--          self._output:cmul(self._repeat)
--       else
--          self._output:cmul(self._expand)
--       end
--    end
   
--    return self.output
-- end

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
   scale = scale or 1
   
   self._input = self._input or input.new()
   self._gradWeight = self._gradWeight or input.new()
   self._sum = self._sum or input.new()
   
   if self.weight:nElement() == gradOutput:nElement() then
      self.gradWeight:addcmul(scale, input, gradOutput)
   else
      local batchSize = input:size(1)
      self._input:view(input, batchSize, -1)
      self._gradOutput:view(gradOutput, batchSize, -1)
      self._gradWeight:view(self.gradWeight, 1, -1)
      
      self._repeat:cmul(self._input, self._gradOutput)
      self._sum:sum(self._repeat, 1)
      self._gradWeight:add(scale, self._sum)
   end
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


