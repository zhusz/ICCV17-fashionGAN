local checkVar = {}

function checkVar.locals()
  local variables = {}
  local idx = 1
  while true do
    local ln, lv = debug.getlocal(2, idx)
    if ln ~= nil then
      variables[ln] = lv
    else
      break
    end
    idx = 1 + idx
  end
  return variables
end

function checkVar.upvalues()
  local variables = {}
  local idx = 1
  local func = debug.getinfo(2, "f").func
  while true do
    local ln, lv = debug.getupvalue(func, idx)
    if ln ~= nil then
      variables[ln] = lv
    else
      break
    end
    idx = 1 + idx
  end
  return variables
end

function checkVar.getKeys(tab)
  local keyset={}
  local n=0

  for k,v in pairs(tab) do
    n=n+1
    keyset[n]=k
  end
  return keyset
end

function checkVar.globals()
  return checkVar.getKeys(_G)
end

return checkVar
