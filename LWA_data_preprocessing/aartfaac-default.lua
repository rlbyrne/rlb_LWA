--[[
 This is the default Aartfaac strategy, which is based on the
 generic "minimal" AOFlagger strategy, version 2020-06-14
 Author: André Offringa
]]--

aoflagger.require_min_version("3.0")

function execute(input)

  --
  -- Generic settings
  --

  local base_threshold = 1.0  -- lower means more sensitive detection
  -- How to flag complex values, options are: phase, amplitude, real, imaginary, complex
  local representation = "amplitude"
  local iteration_count = 3  -- how many iterations to perform?
  local threshold_factor_step = 2.0 -- How much to increase the sensitivity each iteration?
  local frequency_resize_factor = 1.0 -- Amount of "extra" smoothing in frequency direction
  local transient_threshold_factor = 1.0 -- decreasing this value makes detection of transient RFI more aggressive
 
  --
  -- End of generic settings
  --

  local inpPolarizations = input:get_polarizations()

  input:clear_mask()
  
  for ipol,polarization in ipairs(inpPolarizations) do
 
    local converted_data =
        input:convert_to_polarization(polarization):convert_to_complex(representation)
    
    local converted_copy = converted_data:copy()

    for i=1,iteration_count-1 do
      local threshold_factor = math.pow(threshold_factor_step, iteration_count-i)

      local sumthr_level = threshold_factor * base_threshold
      aoflagger.sumthreshold(converted_data, sumthr_level, sumthr_level*transient_threshold_factor, true, true)
 
      -- Do timestep & channel flagging
      local chdata = converted_data:copy()
      aoflagger.threshold_timestep_rms(converted_data, 3.5)
      aoflagger.threshold_channel_rms(chdata, 3.0 * threshold_factor, true)
      converted_data:join_mask(chdata)

      -- High pass filtering steps
      converted_data:set_visibilities(converted_copy)
      local resized_data = aoflagger.downsample(converted_data, 3, frequency_resize_factor, true)
      aoflagger.low_pass_filter(resized_data, 21, 31, 2.5, 5.0)
      aoflagger.upsample(resized_data, converted_data, 3, frequency_resize_factor)

      local tmp = converted_copy - converted_data
      tmp:set_mask(converted_data)
      converted_data = tmp

      aoflagger.set_progress((ipol-1)*iteration_count+i, #inpPolarizations*iteration_count )
    end -- end of iterations

    aoflagger.sumthreshold(converted_data, base_threshold, base_threshold*transient_threshold_factor, true, true)

    if input:is_complex() then
      converted_data = converted_data:convert_to_complex("complex")
    end
    input:set_polarization_data(polarization, converted_data)

    aoflagger.set_progress(ipol, #inpPolarizations )
  end -- end of polarization iterations

  aoflagger.scale_invariant_rank_operator(input, 0.2, 0.2)
  aoflagger.threshold_timestep_rms(input, 4.0)
  input:flag_nans()
end

