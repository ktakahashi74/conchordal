use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::Path;

use conchordal::life::scenario::{BrainConfig, SpawnMethod, TimbreGenotype};
use schemars::schema::{
    InstanceType, RootSchema, Schema, SchemaObject, SingleOrVec, SubschemaValidation,
};
use schemars::schema_for;

fn main() -> Result<(), Box<dyn Error>> {
    let out_dir = Path::new("docs/schemas");
    fs::create_dir_all(out_dir)?;

    write_schema_markdown("brain_config", &schema_for!(BrainConfig), out_dir)?;
    write_schema_markdown("spawn_method", &schema_for!(SpawnMethod), out_dir)?;
    write_schema_markdown("timbre_genotype", &schema_for!(TimbreGenotype), out_dir)?;

    Ok(())
}

fn write_schema_markdown(
    name: &str,
    root: &RootSchema,
    out_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let mut output = String::new();
    let root_schema = Schema::Object(root.schema.clone());
    append_schema_section(&mut output, "Root", &root_schema);

    let defs: BTreeMap<String, Schema> = root
        .definitions
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect();

    for (def_name, schema) in defs.iter() {
        output.push('\n');
        output.push_str(&format!("## {def_name}\n\n"));
        append_schema_section(&mut output, def_name, schema);
    }

    let path = out_dir.join(format!("{name}.md"));
    fs::write(path, output)?;
    Ok(())
}

fn append_schema_section(output: &mut String, label: &str, schema: &Schema) {
    match schema {
        Schema::Bool(_) => {
            output.push_str(&format!("{label}: any\n"));
        }
        Schema::Object(obj) => {
            if let Some(table) = schema_object_table(obj) {
                output.push_str(&table);
                return;
            }

            if let Some(subschemas) = &obj.subschemas {
                if append_variant_tables(output, subschemas) {
                    return;
                }
            }

            let summary = schema_object_summary(obj);
            output.push_str(&format!("{summary}\n"));
        }
    }
}

fn schema_object_table(obj: &SchemaObject) -> Option<String> {
    let properties = obj
        .object
        .as_ref()
        .map(|o| &o.properties)
        .filter(|props| !props.is_empty())?;

    let mut out = String::new();
    out.push_str("| Field | Type | Description |\n");
    out.push_str("| --- | --- | --- |\n");

    let mut keys: Vec<_> = properties.keys().collect();
    keys.sort();

    for key in keys {
        let schema = &properties[key];
        let type_name = schema_type_name(schema);
        let description = schema_description(schema);
        out.push_str(&format!(
            "| {key} | {type_name} | {description} |\n"
        ));
    }

    out.push('\n');
    Some(out)
}

fn append_variant_tables(output: &mut String, subschemas: &SubschemaValidation) -> bool {
    let variants = subschemas
        .one_of
        .as_ref()
        .or(subschemas.any_of.as_ref())
        .filter(|list| !list.is_empty());

    let Some(variants) = variants else {
        return false;
    };

    let mut emitted = false;
    for (idx, schema) in variants.iter().enumerate() {
        let name = variant_name(schema).unwrap_or_else(|| format!("Variant {idx}"));
        output.push_str(&format!("### {name}\n\n"));
        if let Schema::Object(obj) = schema {
            if let Some(table) = schema_object_table(obj) {
                output.push_str(&table);
                emitted = true;
                continue;
            }
        }

        let summary = schema_summary(schema);
        output.push_str(&format!("{summary}\n\n"));
        emitted = true;
    }

    emitted
}

fn schema_object_summary(obj: &SchemaObject) -> String {
    if let Some(values) = &obj.enum_values {
        let items = values
            .iter()
            .filter_map(|v| v.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        if !items.is_empty() {
            return format!("Enum values: {items}");
        }
    }

    if let Some(reference) = &obj.reference {
        return format!("Reference: {reference}");
    }

    "Schema details are complex; see JSON schema.".to_string()
}

fn schema_summary(schema: &Schema) -> String {
    match schema {
        Schema::Bool(_) => "Schema details are complex; see JSON schema.".to_string(),
        Schema::Object(obj) => schema_object_summary(obj),
    }
}

fn schema_type_name(schema: &Schema) -> String {
    match schema {
        Schema::Bool(_) => "any".to_string(),
        Schema::Object(obj) => {
            if let Some(reference) = &obj.reference {
                return reference_name(reference);
            }

            if let Some(instance_type) = &obj.instance_type {
                return match instance_type {
                    SingleOrVec::Single(t) => instance_type_name(t, obj),
                    SingleOrVec::Vec(list) => {
                        let mut names = list
                            .iter()
                            .map(|t| instance_type_name(t, obj))
                            .collect::<Vec<_>>();
                        names.sort();
                        names.join(" or ")
                    }
                };
            }

            if obj.subschemas.is_some() {
                return "enum/union".to_string();
            }

            "object".to_string()
        }
    }
}

fn instance_type_name(instance_type: &InstanceType, obj: &SchemaObject) -> String {
    match instance_type {
        InstanceType::Array => array_item_type(obj).unwrap_or_else(|| "array".to_string()),
        InstanceType::Boolean => "bool".to_string(),
        InstanceType::Integer => "integer".to_string(),
        InstanceType::Null => "null".to_string(),
        InstanceType::Number => "number".to_string(),
        InstanceType::Object => "object".to_string(),
        InstanceType::String => "string".to_string(),
    }
}

fn array_item_type(obj: &SchemaObject) -> Option<String> {
    let items = obj.array.as_ref()?.items.as_ref()?;
    let item_schema = match items {
        SingleOrVec::Single(schema) => schema,
        SingleOrVec::Vec(list) => list.first()?,
    };
    Some(format!("{}[]", schema_type_name(item_schema)))
}

fn schema_description(schema: &Schema) -> String {
    match schema {
        Schema::Bool(_) => String::new(),
        Schema::Object(obj) => obj
            .metadata
            .as_ref()
            .and_then(|m| m.description.as_ref())
            .cloned()
            .unwrap_or_default(),
    }
}

fn reference_name(reference: &str) -> String {
    reference
        .rsplit('/')
        .next()
        .map(|s| s.to_string())
        .unwrap_or_else(|| reference.to_string())
}

fn variant_name(schema: &Schema) -> Option<String> {
    let Schema::Object(obj) = schema else {
        return None;
    };
    let props = &obj.object.as_ref()?.properties;
    let prop = props
        .get("mode")
        .or_else(|| props.get("brain"))
        .or_else(|| props.get("type"))?;
    if let Schema::Object(prop_obj) = prop {
        if let Some(values) = &prop_obj.enum_values {
            if let Some(value) = values.first().and_then(|v| v.as_str()) {
                return Some(value.to_string());
            }
        }
    }
    None
}
