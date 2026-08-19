---
title: Eingabevalidierung
description: Validierung aller Eingaben von Nutzern und externen Systemen.
category:
- Security
- Code
problems:
- sql-injection-vulnerabilities
- cross-site-scripting-vulnerabilities
- buffer-overflow-vulnerabilities
- inadequate-error-handling
- log-injection-vulnerabilities
- integer-overflow-underflow
- silent-data-corruption
- rest-api-design-issues
- null-pointer-dereferences
- entity-attribute-value-overuse
layout: solution
lang: de
en_slug: input-validation
related_solutions:
- slug: canonicalization
  similarity: 0.8
- slug: authentication
  similarity: 0.75
- slug: negative-testing
  similarity: 0.75
- slug: output-encoding
  similarity: 0.75
- slug: value-range-definition
  similarity: 0.75
- slug: data-flow-control
  similarity: 0.7
---

## Description

Eingabevalidierung prüft, dass in ein System eintretende Daten — über Webformulare, API-Aufrufe, Datei-Uploads oder Nachrichten von anderen Systemen — Typ, Länge, Bereich und Format entsprechen, bevor auf diesen Daten gehandelt wird, idealerweise mittels eines Allowlist-Ansatzes, der definiert, was akzeptiert wird, statt einer Denylist, die versucht aufzuzählen, was abgelehnt wird, da Denylists strukturell unvollständig gegenüber neuen Kodierungstricks sind. Legacy-Systeme sind hier überproportional exponiert, weil viele Einstiegspunkte zu einer Zeit gebaut wurden, als das Vertrauen in Eingaben von Nutzern und anderen Systemen die Standardannahme war statt eine explizite Designentscheidung, wodurch String-Verkettung in SQL-Abfragen, ungeprüfte Datei-Uploads und unvalidierte numerische Felder über Dutzende oder Hunderte von Endpunkten verstreut zurückblieben, die sich über die Lebensdauer des Systems angesammelt haben. Validierung in ein solches System nachzurüsten ist notwendigerweise inkrementell und Einstiegspunkt für Einstiegspunkt, und sie muss serverseitig durchgesetzt werden, unabhängig von etwaigen clientseitigen Prüfungen, da clientseitige Validierung eine Bequemlichkeit ist, die jeder Angreifer einfach umgehen kann. Eingabevalidierung ist auch explizit eine Ergänzung zu, nicht ein Ersatz für, strukturelle Verteidigungen wie parametrisierte Abfragen — die beiden zusammen bieten Verteidigung in der Tiefe, wobei parametrisierte Abfragen SQL Injection auf architektonischer Ebene beseitigen und Validierung fehlerhafte oder bösartige Eingabe an der Grenze abfängt, bevor sie überhaupt nachgelagerte Logik erreicht. Die laufenden Kosten sind, dass Validierungsregeln sich zusammen mit Geschäftsanforderungen weiterentwickeln müssen, da übermäßig strikte Regeln legitime Randfälle wie gültige internationale Zeichen ablehnen, während veraltete Regeln neu entdeckte Angriffsmuster nicht fangen.

## How to Apply ◆

> Legacy-Systeme vertrauen häufig Eingaben von Nutzern und externen Systemen ohne Validierung, was Schwachstellen von Injection-Angriffen bis Datenbeschädigung schafft. Umfassende Eingabevalidierung stellt sicher, dass alle in das System eintretenden Daten erwarteten Formaten, Typen und Bereichen entsprechen.

- Identifizieren Sie alle Eingabeeinstiegspunkte: Webformulare, API-Endpunkte, Datei-Uploads, Kommandozeilenargumente, Umgebungsvariablen, Datenbankeingaben von anderen Systemen und Message-Queue-Payloads. Jeder Einstiegspunkt ist ein potenzieller Angriffsvektor.
- Implementieren Sie Allowlist-Validierung (definieren, was akzeptiert wird) statt Denylist-Validierung (definieren, was abgelehnt wird). Denylists sind inhärent unvollständig und können mit neuen Kodierungstricks umgangen werden, während Allowlists den akzeptablen Eingaberaum explizit definieren.
- Validieren Sie Eingabetyp, -länge, -bereich und -format an jedem Einstiegspunkt. Numerische Felder sollten nicht-numerische Eingabe ablehnen, String-Felder sollten Maximallängen durchsetzen, Datumsfelder sollten gültige Datumsformate verifizieren, und aufzählbare Felder sollten nur gültige Werte akzeptieren.
- Wenden Sie Validierung serverseitig an, selbst wenn clientseitige Validierung existiert. Clientseitige Validierung ist eine Bequemlichkeit für die Nutzererfahrung, die trivial umgangen werden kann — serverseitige Validierung ist die Sicherheitskontrolle.
- Nutzen Sie parametrisierte Abfragen oder Prepared Statements für alle Datenbankoperationen, um SQL Injection zu verhindern. Dies ist die effektivste Verteidigung unabhängig von Eingabevalidierung, da sie Code strukturell von Daten trennt.
- Validieren Sie Datei-Uploads, indem Sie den Dateityp prüfen (Magic Bytes, nicht nur Dateiendung), Größenbegrenzungen durchsetzen und auf bösartigen Inhalt scannen. Speichern Sie hochgeladene Dateien außerhalb des Web-Roots mit randomisierten Namen.
- Implementieren Sie strukturiertes Logging, das Log Injection verhindert, indem Sonderzeichen in Log-Einträgen kodiert werden. Angreifer, die Zeilenumbrüche und Steuerzeichen in Logs injizieren können, können Log-Einträge fälschen und ihre Aktivitäten verschleiern.

## Tradeoffs ⇄

> Eingabevalidierung verhindert eine breite Palette von Injection- und Datenbeschädigungsangriffen an der Systemgrenze, erfordert aber umfassende Abdeckung und laufende Pflege, während sich Eingabeanforderungen weiterentwickeln.

**Vorteile:**

- Verhindert Injection-Angriffe (SQL, XSS, Command Injection, LDAP Injection), indem sichergestellt wird, dass Eingabe keinen ausführbaren Code oder Steuerzeichen enthalten kann.
- Fängt fehlerhafte Daten an der Systemgrenze ab, bevor sie Fehler, Beschädigung oder unerwartetes Verhalten in der nachgelagerten Verarbeitung verursachen.
- Verbessert die Datenqualität, indem Format- und Bereichsbeschränkungen durchgesetzt werden, die Legacy-Systemen oft fehlen.
- Verringert die Angriffsfläche, indem Eingabe abgelehnt wird, die nicht bekannt-guten Mustern entspricht, bevor sie die Anwendungslogik erreicht.

**Kosten und Risiken:**

- Umfassende Eingabevalidierung über alle Einstiegspunkte eines Legacy-Systems hinweg erfordert erheblichen Entwicklungsaufwand, besonders wenn Einstiegspunkte zahlreich und verstreut sind.
- Übermäßig strikte Validierung kann legitime Eingabe ablehnen, besonders für internationale Zeichen, ungewöhnliche aber gültige Formate und bei der Implementierung nicht antizipierte Randfälle.
- Validierungsregeln müssen gepflegt werden, während sich Geschäftsanforderungen ändern — veraltete Regeln können neue gültige Eingaben blockieren oder neue ungültige nicht fangen.
- Eingabevalidierung allein verhindert nicht alle Injection-Angriffe — sie muss mit Output-Encoding, parametrisierten Abfragen und anderen Verteidigung-in-der-Tiefe-Maßnahmen kombiniert werden.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Eingabevalidierung Angriffe und Datenbeschädigung in Legacy-Systemen verhindert.

Eine Legacy-Webanwendung konstruiert SQL-Abfragen durch Verkettung von Nutzereingabe direkt in Abfragezeichenfolgen. Ein Angreifer gibt `' OR 1=1 --` in das Benutzernamensfeld ein und erhält Zugriff auf alle Nutzerkonten. Die sofortige Korrektur ersetzt String-Verkettung durch parametrisierte Abfragen in der gesamten Datenzugriffsschicht. Zusätzlich implementiert das Team Eingabevalidierung, die Benutzernamen auf alphanumerische Zeichen und Unterstriche mit einer Maximallänge von 50 Zeichen beschränkt. Die Kombination aus parametrisierten Abfragen (die SQL Injection strukturell verhindern) und Eingabevalidierung (die offensichtlich bösartige Eingabe an der Grenze ablehnt) bietet Verteidigung in der Tiefe. Das Team erweitert dieses Muster auf alle 87 Formularfelder in der Legacy-Anwendung und definiert Validierungsregeln für jedes basierend auf dem erwarteten Datentyp und Format.

Ein Legacy-Auftragsverarbeitungssystem akzeptiert XML-Dateien von Lieferanten per FTP. Eine fehlerhafte XML-Datei mit einem extrem großen Element (5 GB wiederholter Zeichen) lässt den XML-Parser den gesamten verfügbaren Speicher belegen, was den Auftragsverarbeitungsdienst zum Absturz bringt. Das Team implementiert Eingabevalidierung an der Datei-Upload-Grenze: Dateien werden auf 100 MB begrenzt, die XML-Struktur wird vor dem vollständigen Parsen gegen ein Schema validiert, Element- und Attributwerte werden auf definierte Maximallängen begrenzt, und Entity-Expansion wird deaktiviert, um XML-Bomben-Angriffe zu verhindern. Diese Grenzvalidierungen werden in einem Vorverarbeitungsschritt implementiert, der vor dem Legacy-XML-Parser läuft, und schützen ihn vor Eingaben, die Abstürze oder Ressourcenerschöpfung auslösen würden.
