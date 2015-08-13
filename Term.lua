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
   return self.output
end

function Term:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   self.gradWeight:add(-1*scale, gradOutput)
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
   local values = ''
   for i=1,self.size do
      values = values .. string.format("%f", self.weight[i][1])
      if i < self.size then values = values .. ', ' end
   end
   return "Term(" .. values .. ")"
end
