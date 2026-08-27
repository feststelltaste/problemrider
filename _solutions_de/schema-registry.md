---
title: Schema Registry
description: Zentrale Verwaltung von Schemata mit erzwungener
  Datenvertrags-Kompatibilität über Services hinweg.
category:
- Architecture
- Dependencies
problems:
- api-versioning-conflicts
- breaking-changes
- schema-evolution-paralysis
- poor-interfaces-between-applications
- integration-difficulties
- cross-system-data-synchronization-problems
- shared-dependencies
layout: solution
lang: de
en_slug: schema-registry
related_solutions:
- slug: semantic-versioning
  similarity: 0.7
- slug: event-driven-integration
  similarity: 0.7
- slug: standardized-protocols
  similarity: 0.7
- slug: version-control
  similarity: 0.7
- slug: versioning-scheme
  similarity: 0.7
- slug: service-mesh
  similarity: 0.7
---

## Description

Eine Schema Registry ist ein zentraler Dienst, der die Datenverträge — typischerweise Avro-, Protobuf- oder JSON-Schema-Definitionen —, die Dienste zum Austausch von Nachrichten oder Ereignissen nutzen, speichert, versioniert und Kompatibilitätsregeln für sie durchsetzt, und lehnt Schemaänderungen ab, die bestehende Konsumenten brechen würden, bevor sie je Produktion erreichen. Kompatibilität wird in definierten Modi (rückwärts, vorwärts oder vollständig) als Teil der CI/CD-Pipeline durchgesetzt, sodass eine inkompatible Feldentfernung oder Typänderung zur Build-Zeit erfasst wird, statt später als Laufzeit-Deserialisierungsfehler aufzutauchen. Dies adressiert einen spezifischen Fehlermodus, der in Legacy-Systemen häufig ist, die zu einem Netz von Diensten gewachsen sind, die Daten über informell vereinbarte, undokumentierte Formate austauschen: Da keine einzelne Quelle der Wahrheit für diese Formate existiert, kann die scheinbar harmlose Änderung einer gemeinsamen Ereignisstruktur durch ein Team still mehrere andere Dienste brechen, die nie zur Änderung konsultiert wurden. Die Einführung einer Schema Registry in eine solche Umgebung erfolgt üblicherweise schrittweise, indem zuerst die bestehenden, bereits informellen Verträge als Basislinie registriert werden und dann die gesamte nachfolgende Schemaentwicklung unter die Governance der Registry gestellt wird, statt zu versuchen, jeden Datenvertrag auf einmal neu zu gestalten. Über die Verhinderung brechender Änderungen hinaus wird die Versionshistorie der Registry zu einer Form lebender Dokumentation, wie sich jeder Datenvertrag entwickelt hat, was in Legacy-Umgebungen wertvoll ist, wo die ursprüngliche Begründung für ein gegebenes Feld oder Format sonst verloren ginge.

## How to Apply ◆

- Führen Sie eine zentrale Schema Registry ein (z. B. Confluent Schema Registry, Apicurio), in der alle Serviceverträge gespeichert und versioniert werden.
- Definieren Sie Kompatibilitätsmodi (rückwärts, vorwärts, vollständig) und erzwingen Sie sie als Teil der CI/CD-Pipeline, sodass inkompatible Schemaänderungen vor der Bereitstellung abgelehnt werden.
- Migrieren Sie Legacy-Dienste schrittweise, indem Sie zuerst ihre bestehenden Datenverträge registrieren und dann Schemata unter der Governance der Registry weiterentwickeln.
- Integrieren Sie Schemavalidierung in Produzenten- und Konsumentendienste, sodass Serialisierungs-/Deserialisierungsfehler früh erfasst werden.
- Etablieren Sie Eigentumsregeln: jedes Schema hat ein designiertes Team, das für seine Weiterentwicklung verantwortlich ist.
- Nutzen Sie die Kompatibilitätsprüfungen der Registry, um manuelles Review von Schnittstellenänderungen an Legacy-Integrationspunkten zu ersetzen.

## Tradeoffs ⇄

**Vorteile:**
- Verhindert, dass brechende Änderungen Produktion erreichen, indem Kompatibilitätsregeln automatisch durchgesetzt werden.
- Bietet eine einzige Quelle der Wahrheit für alle Datenverträge und reduziert Missverständnisse zwischen Teams.
- Macht Schemaentwicklung explizit und auditierbar, was Compliance und Debugging erleichtert.
- Reduziert Integrationsfehler, wenn mehrere Legacy-Dienste Datenformate teilen.

**Kosten:**
- Fügt Infrastrukturkomplexität hinzu; die Registry selbst wird zu einer Abhängigkeit, die betrieben und überwacht werden muss.
- Erfordert anfänglichen Aufwand, um bestehende Schemata aus Legacy-Systemen zu katalogisieren und zu registrieren.
- Strenge Kompatibilitätsmodi können Schemaentwicklung verlangsamen, wenn schnelle Änderungen benötigt werden.
- Teams müssen neues Tooling lernen und ihre Entwicklungsabläufe anpassen.

## How It Could Be

Ein Finanzdienstleistungsunternehmen betreibt einen Legacy-Message-Broker, in dem zwölf Dienste Avro-kodierte Ereignisse austauschen. Nach mehreren Produktionsvorfällen, verursacht durch unkoordinierte Schemaänderungen, setzt es eine Schema Registry ein und registriert alle bestehenden Schemata. Der CI-Build jedes Dienstes validiert nun neue Schemaversionen gegen die Rückwärtskompatibilitätsregeln der Registry. Innerhalb von drei Monaten sinken integrationsbezogene Vorfälle erheblich, weil inkompatible Änderungen zur Build-Zeit statt zur Laufzeit erfasst werden. Teams, die zuvor Stunden mit dem Debuggen von Deserialisierungsfehlern verbrachten, können sich nun auf Feature-Arbeit konzentrieren, und die Versionshistorie der Registry dient als lebende Dokumentation, wie sich Datenverträge über die Jahre entwickelt haben.
