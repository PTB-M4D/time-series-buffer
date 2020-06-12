{{summaryPlaceholder}}

{{#parametersExist}}
Parameters
----------
{{/parametersExist}}
{{#args}}
    {{var}}: {{typePlaceholder}}
        {{descriptionPlaceholder}}
{{/args}}
{{#kwargs}}
    {{var}}: {{typePlaceholder}} (default: {{&default}})
        {{descriptionPlaceholder}}
{{/kwargs}}

{{#returnsExist}}
Returns
-------
{{/returnsExist}}
{{#returns}}
    {{var}}: {{typePlaceholder}}
        {{descriptionPlaceholder}}
{{/returns}}